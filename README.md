# Photo Geolocator — make.fun bounty entry (photo-geolocator-01)

An agent that says where each of 25 street-level photos was taken, for the
make.fun competition "Find where a photo was taken".

## How it works

`main.py` (Python 3.13, standard library only) runs one trial:

1. It fetches the instance's 25 photos from the env at `MAKEFUN_ENV_URL`.
2. **Pass 1** — one `claude-sonnet-5-5` call per photo through the trial
   gateway, with a structured GeoGuessr prompt: the model must reason step
   by step over the visible evidence (text and language, driving side,
   road markings, architecture, vegetation, vehicles, landmarks) and narrow
   continent → country → region → city before giving one JSON object with
   `lat`, `lon`, `country`, `place` and a `confidence` 0–1. Four calls run
   at a time; answers are collected by photo id, never by completion order.
3. **Pass 2** — the 8 lowest-confidence photos get a second look: a
   `claude-opus-5-5` call that shows the photo again with pass 1's estimate
   and asks the model to confirm or correct it. A refused opus call falls
   back to a sonnet second look for that photo. The second answer replaces
   the first only when its confidence is at least the first one's.
4. **Vendor fallback** — a photo whose Anthropic call fails is tried once
   on OpenAI's `gpt-6-sol` with the same prompt. A vendor that answers 401
   (no key pushed for it) is remembered as dead for the rest of the trial.
5. **Fallback** — a photo no model call could place still gets an answer:
   one fixed prior coordinate, the point minimising the mean great-circle
   distance to the 75 photos of the kit's public practice pool (a single
   aggregate statistic of public practice data, not a table of photos and
   their places). An unanswered photo scores the maximum error, so a prior
   guess always beats leaving it blank.
6. It posts all 25 answers to the env in photo order and exits.

## Determinism

No wall clock and no unseeded randomness: every request is a function of
the photo bytes, the fixed prompts and the answers before it. Retries only
happen for a response that actually arrived (408, 429, 5xx, 529), with
fixed sleeps, at most twice; a gateway refusal is final, and a connection
that failed before any answer is never retried. Two local runs on practice
trial-0 left byte-identical `state.json` files.

## Why this beats the house baseline

The baseline is one plain `claude-sonnet-5-5` call per photo ("where was
this photo taken?") with no reasoning prompt and no second look. This
entry adds a structured evidence-first prompt, a stronger model
(`claude-opus-5-5`) for a second look at the hardest photos, a second
vendor when Anthropic has no key, and a prior so no photo is ever left
unanswered.

## Model keys

Model calls run on the entrant's own Anthropic or OpenAI key, pushed to
the competition before submitting (`makefun secrets push
photo-geolocator-01 --env-file .env` with `ANTHROPIC_API_KEY` and/or
`OPENAI_API_KEY`). With no key pushed, every call is answered 401 and the
entry scores its prior only. The house pays for none of an entry's calls.

## Files

- `main.py` — the agent (the entrypoint the grader runs)
- `README.md` — this file
- `LICENSE` — MIT
- `.makefun/owner.json` — claim file naming the linked Solana wallet
