#!/usr/bin/env python3
"""photo-geolocator-01 entry: a vision-model geolocator, Python 3.13, stdlib only.

Reads the instance's 25 street-level photos from the env at MAKEFUN_ENV_URL,
places each one with an allowlisted vision model through the trial gateway,
and posts latitude/longitude answers back to the env. No network besides the
env and the gateway; no wall clock and no unseeded randomness, so the live
pass and the replay send the same calls and leave the same answers.

How it places a photo
---------------------
Pass 1 -- one claude-sonnet-5-5 call per photo with a structured GeoGuessr
prompt: the model must reason step by step over the visible evidence (text
and language, driving side, road markings, architecture, vegetation,
vehicles, landmarks) and narrow continent -> country -> region -> city
before giving one JSON object with lat, lon, country, place and a
confidence 0..1. Four calls run at a time; answers are collected by photo
id, never by completion order.

Pass 2 -- the 8 lowest-confidence photos get a second look: a
claude-opus-5-5 call that shows the photo again with pass 1's estimate and
asks the model to confirm or correct it. A refused opus call falls back to
a sonnet second look for that photo. The second answer replaces the first
only when its confidence is at least the first one's.

Vendor fallback -- a photo whose Anthropic call fails (401: no Anthropic
key was pushed for this competition, or any other final error) is tried
once on OpenAI's gpt-6-sol with the same prompt. A vendor that answers
401 is remembered as dead for the rest of the trial, so no time is spent
re-asking it.

Fallback -- a photo no model call could place still gets an answer: one
fixed prior coordinate, the point that minimises the mean great-circle
distance to the 75 photos of the kit's public practice pool (a single
aggregate statistic of public practice data, not a table of photos and
their places). An unanswered photo scores the maximum error, so a prior
guess always beats leaving it blank.

Retries -- only a response that actually arrived is retried (408, 429,
5xx, 529), with fixed sleeps, at most twice. A gateway refusal (a 4xx
whose error type is budget_exhausted, request_cap, model_not_allowed or
max_tokens_required, or x-should-retry: false) is final, as is a 401.
A connection that failed before any answer is never retried: the replay
has no recording of a request the gateway never saw.
"""

import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- constants

MODEL_PASS1 = "claude-sonnet-5-5"
MODEL_REFINE = "claude-opus-5-5"
MODEL_OPENAI = "gpt-6-sol"
MAX_TOKENS = 1536
CONCURRENCY = 4
REFINE_COUNT = 8
MAX_RETRIES = 2
RETRY_SLEEPS_S = (2.0, 4.0)
RETRY_STATUSES = (408, 429, 500, 502, 503, 504, 529)
GATEWAY_REFUSALS = ("budget_exhausted", "request_cap", "model_not_allowed",
                    "max_tokens_required")
HTTP_TIMEOUT_S = 120

# One fixed prior: the geometric-median-ish point of the kit's 75 public
# practice photos (mean error 6,843 km over them, vs 8,407 km at 0,0).
# A single aggregate number, not a lookup of any photo's place.
FALLBACK_LAT = 56.71875
FALLBACK_LON = 27.03125

PROMPT1 = (
    "You are an expert GeoGuessr player. Look carefully at this street-level "
    "photo and work out where on Earth it was taken. Reason step by step "
    "about the visible evidence, then give your single best estimate:\n"
    "1. Text: signs, shop names, number plates, road signs - language, "
    "script, and what they say.\n"
    "2. Driving: side of the road, road markings and their colours, "
    "bollards, street furniture.\n"
    "3. Buildings and infrastructure: architecture, roof shapes, power "
    "lines, pavement, fences.\n"
    "4. Nature: vegetation, trees, crops, terrain, climate, quality of "
    "light.\n"
    "5. Vehicles: models, plate colours, which side the steering wheel is "
    "on.\n"
    "6. Landmarks: anything distinctive, flags, place names.\n"
    "Narrow it down: continent, country, region, then city or town, then "
    "road if you can. Reply with exactly one JSON object and nothing else: "
    '{"lat": <decimal degrees north>, "lon": <decimal degrees east>, '
    '"country": "<country name>", "place": "<city, region or unknown>", '
    '"confidence": <0 to 1>, "reasoning": "<one or two sentences>"}'
)

PROMPT2 = (
    "Earlier you placed this photo at latitude {lat}, longitude {lon} "
    "({country}, {place}) with confidence {conf}. Look at the photo again "
    "very carefully and check that estimate against the visible evidence - "
    "signs and language, road markings, driving side, vegetation, "
    "architecture, vehicles. Either confirm the estimate or correct it. "
    "Reply with exactly one JSON object and nothing else: "
    '{"lat": <decimal degrees north>, "lon": <decimal degrees east>, '
    '"country": "<country name>", "place": "<city, region or unknown>", '
    '"confidence": <0 to 1>, "reasoning": "<one or two sentences>"}'
)

# ---------------------------------------------------------------- small http

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _NoAnswer(Exception):
    """The exchange failed before any HTTP answer arrived."""


def _exchange(method, url, headers, body):
    """One HTTP exchange -> (status, body bytes). Raises _NoAnswer."""
    req = urllib.request.Request(url, data=body, method=method,
                                 headers=headers)
    try:
        with _OPENER.open(req, timeout=HTTP_TIMEOUT_S) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        finally:
            err.close()
    except (urllib.error.URLError, OSError) as err:
        raise _NoAnswer(str(getattr(err, "reason", err))) from None


def _json_body(status, raw, what):
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise ValueError(f"{what} answered {status} with a body that is "
                         f"not JSON: {raw[:200]!r}") from None


def _error_type(raw):
    """The gateway/vendor error type in a body, or None."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        t = err.get("type") or err.get("code")
        return t if isinstance(t, str) else None
    return None


# ---------------------------------------------------------------- model calls

class _Vendor:
    """One model vendor through the gateway, with a dead latch (401)."""

    def __init__(self, name, base_url, api_key):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dead = False  # latched on 401: no key for this vendor

    def _call_once(self, route, headers, body):
        return _exchange("POST", self.base_url + route, headers, body)

    def ask(self, model, prompt, jpeg_b64, route_kind):
        """One model call -> answer text, or None when it finally failed.

        route_kind is 'anthropic' (Messages) or 'openai' (Chat Completions).
        Retries only answers that arrived with a retryable status.
        """
        if self.dead or not self.api_key:
            return None
        if route_kind == "anthropic":
            route = "/v1/messages"
            headers = {"Content-Type": "application/json",
                       "Accept": "application/json",
                       "x-api-key": self.api_key,
                       "anthropic-version": "2023-06-01"}
            payload = {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg",
                                "data": jpeg_b64}},
                    {"type": "text", "text": prompt}]}],
            }
        else:
            route = "/chat/completions"
            headers = {"Content-Type": "application/json",
                       "Accept": "application/json",
                       "Authorization": "Bearer " + self.api_key}
            payload = {
                "model": model,
                "max_completion_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64,"
                                   + jpeg_b64}}]}],
            }
        body = json.dumps(payload, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        for attempt in range(MAX_RETRIES + 1):
            try:
                status, raw = self._call_once(route, headers, body)
            except _NoAnswer as err:
                sys.stderr.write(f"{self.name}: no answer for {model}: "
                                 f"{err}\n")
                return None  # never retried: the gateway never saw it
            if 200 <= status < 300:
                try:
                    msg = _json_body(status, raw, f"{self.name} {model}")
                except ValueError as err:
                    sys.stderr.write(f"{self.name}: {err}\n")
                    return None
                if route_kind == "anthropic":
                    parts = [b.get("text", "") for b in
                             msg.get("content", [])
                             if isinstance(b, dict) and b.get("type") == "text"]
                    return "\n".join(parts)
                choices = msg.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    content = (choices[0].get("message") or {}).get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return "\n".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict))
                return None
            if status == 401:
                self.dead = True
                sys.stderr.write(f"{self.name}: HTTP 401 for {model} "
                                 f"(no key for this vendor)\n")
                return None
            err_type = _error_type(raw)
            retryable = (status in RETRY_STATUSES
                         and err_type not in GATEWAY_REFUSALS)
            sys.stderr.write(f"{self.name}: {model} refused (HTTP {status}"
                             f"{' ' + err_type if err_type else ''})\n")
            if not retryable or attempt >= MAX_RETRIES:
                return None
            time.sleep(RETRY_SLEEPS_S[attempt])
        return None


# ---------------------------------------------------------------- parsing

def _is_lat(v):
    return (type(v) in (int, float) and v == v and -90.0 <= v <= 90.0)


def _is_lon(v):
    return (type(v) in (int, float) and v == v and -180.0 <= v <= 180.0)


def parse_guess(text):
    """A model's answer text -> {lat, lon, country, place, confidence} or None.

    First JSON object with lat and lon wins; else the first two numbers in
    range. Confidence defaults to 0 when absent or unusable.
    """
    if not text:
        return None
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        lat = obj.get("lat", obj.get("latitude"))
        lon = obj.get("lon", obj.get("lng", obj.get("longitude")))
        if _is_lat(lat) and _is_lon(lon):
            conf = obj.get("confidence", 0.0)
            if type(conf) not in (int, float) or conf != conf:
                conf = 0.0
            conf = max(0.0, min(1.0, float(conf)))
            country = obj.get("country", "")
            place = obj.get("place", "")
            return {"lat": float(lat), "lon": float(lon),
                    "country": country if isinstance(country, str) else "",
                    "place": place if isinstance(place, str) else "",
                    "confidence": conf}
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
    for i in range(len(nums) - 1):
        if _is_lat(nums[i]) and _is_lon(nums[i + 1]):
            return {"lat": nums[i], "lon": nums[i + 1], "country": "",
                    "place": "", "confidence": 0.0}
    return None


# ---------------------------------------------------------------- env

def _get_json(url):
    status, raw = _exchange("GET", url, {"Accept": "application/json"}, None)
    if status != 200:
        raise RuntimeError(f"GET {url} answered {status}: {raw[:200]!r}")
    return _json_body(status, raw, f"GET {url}")


def _get_bytes(url):
    status, raw = _exchange("GET", url, {}, None)
    if status != 200:
        raise RuntimeError(f"GET {url} answered {status}: {raw[:200]!r}")
    return raw


def post_answers(env_url, answers):
    """POST the answers (one POST, retries only on an answer that arrived
    with a retryable status). Returns True when the env accepted them."""
    body = json.dumps({"answers": answers}, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        try:
            status, raw = _exchange(
                "POST", env_url + "/answers",
                {"Content-Type": "application/json",
                 "Accept": "application/json"}, body)
        except _NoAnswer as err:
            sys.stderr.write(f"POST /answers: no answer: {err}\n")
            return False
        if 200 <= status < 300:
            sys.stderr.write(f"POST /answers: HTTP {status} "
                             f"{raw[:200].decode('utf-8', 'replace')}\n")
            return True
        if status == 409:
            # Already answered: an earlier POST of ours landed.
            sys.stderr.write("POST /answers: HTTP 409 (already answered)\n")
            return True
        sys.stderr.write(f"POST /answers: HTTP {status} "
                         f"{raw[:200].decode('utf-8', 'replace')}\n")
        if status not in RETRY_STATUSES or attempt >= MAX_RETRIES:
            return False
        time.sleep(RETRY_SLEEPS_S[attempt])
    return False


# ---------------------------------------------------------------- main

def main():
    env_url = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    anthropic = _Vendor("anthropic",
                        os.environ.get("ANTHROPIC_BASE_URL",
                                       "https://api.anthropic.com"),
                        os.environ.get("ANTHROPIC_API_KEY", ""))
    openai = _Vendor("openai",
                     os.environ.get("OPENAI_BASE_URL",
                                    "https://api.openai.com/v1"),
                     os.environ.get("OPENAI_API_KEY", ""))
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            f.read()  # the brief; the env is the only input that matters

    listed = _get_json(env_url + "/photos")
    photos = [p for p in listed["photos"] if not p.get("answered")]
    photos.sort(key=lambda p: p["id"])
    n = len(photos)
    sys.stderr.write(f"geolocator: {n} photos to place\n")

    # Fetch every photo once; a photo that fails to fetch gets the fallback.
    jpegs = {}
    for p in photos:
        try:
            jpegs[p["id"]] = _get_bytes(env_url + p["url"])
        except (RuntimeError, _NoAnswer) as err:
            sys.stderr.write(f"geolocator: cannot fetch {p['id']}: {err}\n")

    guesses = [None] * n  # indexed like photos; never by completion order

    def place_one(i):
        pid = photos[i]["id"]
        jpeg = jpegs.get(pid)
        if jpeg is None:
            return
        b64 = base64.b64encode(jpeg).decode("ascii")
        text = anthropic.ask(MODEL_PASS1, PROMPT1, b64, "anthropic")
        if text is None:
            text = openai.ask(MODEL_OPENAI, PROMPT1, b64, "openai")
        g = parse_guess(text)
        if g is None:
            sys.stderr.write(f"geolocator: no coordinates for {pid}\n")
        guesses[i] = g

    # Pass 1, four workers, results by index.
    next_i = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = next_i[0]
                next_i[0] += 1
            if i >= n:
                return
            place_one(i)

    threads = [threading.Thread(target=worker)
               for _ in range(min(CONCURRENCY, n))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Pass 2: second look at the least confident photos, opus first,
    # sonnet on a refused opus call; keep the second answer only when its
    # confidence is at least the first one's.
    order = sorted(
        range(n),
        key=lambda i: ((guesses[i] or {}).get("confidence", 0.0),
                       photos[i]["id"]))
    refine_idx = [i for i in order
                  if guesses[i] is not None][:REFINE_COUNT]
    sys.stderr.write(f"geolocator: refining "
                     f"{[photos[i]['id'] for i in refine_idx]}\n")

    def refine_one(i):
        pid = photos[i]["id"]
        jpeg = jpegs.get(pid)
        if jpeg is None:
            return
        g1 = guesses[i]
        b64 = base64.b64encode(jpeg).decode("ascii")
        # Plain .replace, not str.format: PROMPT2's JSON braces would be
        # read as format fields.
        prompt2 = (PROMPT2.replace("{lat}", repr(g1["lat"]))
                   .replace("{lon}", repr(g1["lon"]))
                   .replace("{country}", g1["country"] or "unknown")
                   .replace("{place}", g1["place"] or "unknown")
                   .replace("{conf}", repr(g1["confidence"])))
        text = anthropic.ask(MODEL_REFINE, prompt2, b64, "anthropic")
        if text is None and not anthropic.dead:
            # Opus refused but Anthropic is alive: sonnet second look.
            text = anthropic.ask(MODEL_PASS1, prompt2, b64, "anthropic")
        if text is None:
            text = openai.ask(MODEL_OPENAI, prompt2, b64, "openai")
        g2 = parse_guess(text)
        if g2 is not None and g2["confidence"] >= g1["confidence"]:
            guesses[i] = g2
            sys.stderr.write(f"geolocator: {pid} refined "
                             f"({g1['lat']},{g1['lon']}) -> "
                             f"({g2['lat']},{g2['lon']})\n")

    next_r = [0]

    def rworker():
        while True:
            with lock:
                j = next_r[0]
                next_r[0] += 1
            if j >= len(refine_idx):
                return
            refine_one(refine_idx[j])

    rthreads = [threading.Thread(target=rworker)
                for _ in range(min(CONCURRENCY, len(refine_idx)) or 1)]
    for t in rthreads:
        t.start()
    for t in rthreads:
        t.join()

    answers = []
    for i, p in enumerate(photos):
        g = guesses[i]
        if g is None:
            lat, lon = FALLBACK_LAT, FALLBACK_LON
        else:
            lat, lon = g["lat"], g["lon"]
        answers.append({"id": p["id"],
                        "lat": round(float(lat), 6),
                        "lon": round(float(lon), 6)})
    sys.stderr.write(f"geolocator: posting {len(answers)} answers\n")
    ok = post_answers(env_url, answers)
    sys.stdout.write(f"geolocator: {len(answers)} of {n} photos answered, "
                     f"posted={ok}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyError as err:
        sys.stderr.write(f"geolocator: missing environment variable {err}\n")
        sys.exit(1)
    except (RuntimeError, _NoAnswer, ValueError) as err:
        sys.stderr.write(f"geolocator: {err}\n")
        sys.exit(1)
