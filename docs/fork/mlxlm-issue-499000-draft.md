# Upstream issue draft — mlx-lm BatchGenerator dies at ~10.5K decode steps: `[metal::malloc] Resource limit (499000) exceeded`

Draft for an ml-explore/mlx-lm issue (filing is Tim's call). Everything below
was measured 2026-08-30/31 on the Mac Studio (M1 Ultra 64 GB, macOS, python
3.11). Fork context: PATCHES.md #84.

## Summary

Long batched decode on a hybrid (Gated DeltaNet + attention) model exhausts
Metal's per-process **resource count** — not bytes — after ~10,500 decode
steps. Bytes are flat (16.0 GB active, single sequence) while the resource
count climbs ~47/step until `metal::malloc` refuses at 499000:

```
File ".../mlx_lm/generate.py", line 1419, in _step
    mx.async_eval(self._next_tokens, self._next_logprobs, token_context)
RuntimeError: [metal::malloc] Resource limit (499000) exceeded.
```

## Repro (pure mlx-lm, ~30 lines, no third-party code)

```python
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

model, tokenizer = load("mlx-community/Qwen3.6-27B-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Count upward from 1, one number per line. "
      "Do not stop, do not summarize, do not explain. Just keep counting."}],
    add_generation_prompt=True,
)
gen = BatchGenerator(model, max_tokens=24000)
gen.insert([list(prompt)])
steps, done = 0, False
while not done:
    for r in gen.next():
        if getattr(r, "finish_reason", None):
            done = True
    steps += 1
    if steps % 1024 == 0:
        print(f"step={steps} active={mx.get_active_memory()/1e9:.1f}GB", flush=True)
```

Dies between step 10,240 and 10,752 every time (7/7 runs, T=0 and default
sampling, direct and through a serving layer).

## What it is NOT (probe matrix, all crashing at the same step)

| Variable | Values probed | Crash step moved? |
|---|---|---|
| mlx-lm version | `9acef5f`, `74e7cf9`, `77c33b1` (tips around 2026-08) | no — identical |
| mlx / mlx-metal | 0.32.1, **0.32.2** | no |
| serving layer | vllm-mlx fork (features on), fork (all features off), **pure mlx-lm** | no |
| periodic `mx.clear_cache()` | every ~32 steps / never | no (cache bytes visibly differ; crash identical) |
| periodic `mx.eval` of every reachable array | tokens only / + all cache `.state` arrays / + `offset`/`left_padding`/`lengths` metadata attrs | no |

Additional facts:

- A `gc.get_objects()` census shows the **Python-side mx-array count dead
  flat (2,063)** for the whole run while the resource count climbs — the
  retention is below Python.
- ~47 leaked resources per decode step ≈ the model's layer count; the
  leaked objects are tiny (bytes stay flat to GB resolution), so this looks
  like one small per-layer allocation per step that is never released at
  the Metal level.
- mlx-lm #1632 (`11a6ce7`, "Fix unbounded ArraysCache metadata graph
  during decode") fixed a bug with the SAME error signature; this one
  survives it (and predates it — `9acef5f` crashes identically).
- Single-stream `generate_step` was not probed; the batched path is the
  production path here.

## Impact

Any completion longer than ~10.5K tokens on this model class kills the
request (and without an outer recovery layer, the process). Agentic
workloads with 16K+ max_tokens hit it on long single turns.

## Workaround in our fork

`VLLM_MLX_MAX_COMPLETION_TOKENS` — an admission-time clamp on `max_tokens`
below the measured wall, so requests finish as `length` instead of dying
mid-flight (vllm-mlx fork PATCHES.md #84).
