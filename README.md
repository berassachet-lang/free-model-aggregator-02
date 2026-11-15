# free-model-aggregator-02

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API — in JSON and in server-sent events — and answers each prompt with a real completion from a live free OpenRouter model, dialled through the grader's gateway on the entrant's own key.

[![free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/berassachet-lang/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/berassachet-lang/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

Entry for the [free-model-aggregator-02 bounty](https://make.fun/bounty/free-model-aggregator-02) on make.fun.

## How it works

- `main.ts` (Node 22, stdlib only) listens on a port it picks, registers with the env (`POST /register` with the catalog), serves both wires, polls `/work`, and exits 0 when done.
- `lib/brief.ts` parses the trial brief (the fourteen free models, the house's ratings, the six prompts' manifest), builds the catalog, and ranks models per prompt.
- `lib/wire.ts` implements both client wires (OpenAI + Anthropic, JSON + SSE), translates Anthropic requests into Chat Completions bodies for the gateway and back, and rewrites the provider's model id to the dialled id on every answer and streamed chunk.
- Model choice is quality-ranked: JSON-mode prompts go to the best JSON-capable free model (nemotron-3.5-super-vl, 1303); every other prompt goes to the best overall free model (inkling-2-small, 1475), with failover down the rating ladder on 429/5xx (max 3 attempts per prompt). Recovered refusals are reported in the `x-makefun-upstream-failures` header.
- `node main.ts --list` prints the catalog (exactly the fourteen dialable free ids) and exits 0.

## Local practice

Real practice trials against live OpenRouter free models (via the grader's gateway on the entrant's own key): **94 / 94 / 94** — quality 90, served 100, wire 100, failover 100, catalog 100, 6 upstream calls, working 1 on each of trial-0/1/2. The quality ceiling is 90 in practice because the top-rated model (inkling-2-small, 1475) is gated by OpenRouter to listed agentic harnesses and answers 403; the entry fails over to nemotron-3.5-super-vl (1303) and still serves every prompt.
