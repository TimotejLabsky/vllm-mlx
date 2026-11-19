# Fork vs upstream vllm-mlx — 2026-11-19

The measurement behind the table at the top of the README.

## Setup

- **Box:** Mac Studio M1 Ultra, 64 GB. mlx 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.12 for both engines (one venv).
- **Fork:** `main` `b98dc212`, installed.
- **Upstream:** stock `waybarrios/vllm-mlx` `3c228769` (the commit the fork is rebased on), installed with `pip install --no-deps --target` into a separate dir and put first on `PYTHONPATH`.
- **Command and env:** each route's production `cmd` and `env` from the live llama-swap config, for both engines. Upstream got the same command minus CLI flags it does not have; for these routes that was only `--text-only`. Fork env vars are simply unused by upstream.
- **Isolation:**
  - one server at a time on a spare port, with the live 27B-8bit route paused (`paused2.sh`) and nothing else loaded;
  - the SSD tier pointed at a fresh temp dir per run, so no live cache entries were reused;
  - `HF_HUB_OFFLINE=1`.
- **Requests:** streamed; TTFT and decode rate measured by the client. The decode rate is chunks per second between the first and last streamed chunk, reasoning and tool-call deltas included. Sampling is each route's own defaults; the 27B-8bit is T=1.0 / top_p 0.95 / top_k 20.
- **Harness:** [`scripts/fork/bench_fork_vs_upstream.py`](../../scripts/fork/bench_fork_vs_upstream.py).

## Scenarios

- **decode:** 3 short prompts, 400 tokens. The table reports the median.
- **session:** an agent-shaped job.
  - Setup: a system prompt (~2K tokens) plus 16 tools (4.7K tokens in total with the template), and a user task.
  - Each turn appends the model's tool call and a ~1.2K-token tool result.
  - Measured: TTFT per turn and total wall.
  - Size: 6 turns on the 27B (4.7K→10.8K tokens), 4 on the 35B.
- **depth:** a new user message after the session ("write the full report", 12.1K tokens), 400 tokens. It measures decode at depth, and its TTFT includes the prefix this template re-renders when a new user message follows tool turns.
- **toolrestart:** a 3-turn session, 45 s idle (the SSD writer's window), server restart, then the session's next tool turn.
- **image:** an image and text request, 300 tokens ×3, then a follow-up turn about the same image.

## Results

### Qwen3.8-27B-8bit

| | upstream | fork |
|---|---|---|
| Decode, short prompts (tok/s) | 15.16, 15.11, 15.13 | 24.44, 26.06, 23.10 |
| Session TTFT per turn (s) | 21.43, 5.62, 5.52, 5.58, 5.62, 5.7 | 21.9, 5.68, 5.72, 6.08, 6.16, 6.3 |
| Session wall (s) | 108.1 | 99.3 |
| Depth: TTFT (s) / decode (tok/s) | 53.5 / 14.7 | 35.1 / 22.8 |
| Toolrestart: session TTFT per turn (s) | 21.46, 5.61, 5.66 | 23.95, 5.66, 5.71 |
| Toolrestart: TTFT after restart (s) | 37.86 | 19.75 |

After the restart, the fork restored the session's opening 4,721 tokens from SSD and prefilled 3,482. The two later turns had grown the chain by less than the 16K spill delta (#164), so they had not been re-spilled. Upstream prefilled all 8.2K.

### Qwen3.6-35B-A3B-4bit-DWQ

| | upstream | fork |
|---|---|---|
| Decode, short prompts (tok/s) | 54.60, 54.72, 54.75 | 87.16, 87.82, 87.97 |
| Session TTFT per turn (s) | 3.15, 0.90, 0.94, 0.97 | 3.41, 0.94, 0.97, 0.98 |
| Session wall (s) | 18.2 | 12.8 |
| Next user turn after a restart, TTFT (s) | 6.43 | 4.78 |

The restart scenario on this route used a new user message, not a tool continuation. Its SSD writes had not landed when the server stopped.

### Qwen3-VL-30B-A3B-Instruct-8bit

| | upstream | fork |
|---|---|---|
| Image + text decode (tok/s) | 33.74, 33.67, 33.07 | 57.87, 57.73, 57.55 |
| Image follow-up TTFT (s) | 0.60 | 0.35 |

## Reading it

- **Decode gains are large and consistent:** +55–71% across all three routes.
  - Upstream serves these multimodal-format checkpoints through mlx-vlm, because it has no `--text-only`.
  - The fork serves text through mlx-lm's BatchedEngine.
  - The 27B-8bit adds the MTP drafter with speculative sampling (#160/#161/#167).
  - The Qwen3-VL route loses mlx-vlm's per-layer host sync (#168).
- **Prefill is not faster.** Cold first turns are within noise.
- **Cached tool-loop follow-ups are not faster.** Upstream's prefix cache now hits them too. On the 27B the fork is about 0.5 s later to first token there; the cause has not been investigated.
- **The cache gains show where the prompt is not a plain extension of the previous one:** a new user turn after tool turns (−34%) and a reload (−48%).
- **Not measured here:**
  - re-rendered generated spans (#139, #166) and interleaved clients (#163, #165), which have their own real-server checks in PATCHES.md;
  - memory-ceiling behaviour, also covered in PATCHES.md.
