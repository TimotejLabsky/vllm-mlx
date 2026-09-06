#!/usr/bin/env python3
"""Apply the experimental decode-burst patch to the bench checkout.

VLLM_MLX_DECODE_BURST=K chains up to K scheduler steps per executor hand-off
(oMLX's burst-decode trick: avoid per-token event-loop/GIL ping-pong), merging
the K SchedulerOutputs into one. Bounded by VLLM_MLX_DECODE_BURST_BUDGET_S
(default 0.1s); breaks early on a finished request, an empty step, or an
empty scheduler. Default (unset / 1) is byte-for-byte current behavior.
"""
P = "/Users/ai/vllm-mlx-src/vllm_mlx/engine_core.py"
s = open(P).read()

if "VLLM_MLX_DECODE_BURST" in s:
    print("already patched")
    raise SystemExit(0)

s = s.replace("import asyncio\nimport logging\nimport time",
              "import asyncio\nimport logging\nimport os\nimport time", 1)

old = """        def _step_on_worker():
            _bind_worker_streams_once()
            output = self.scheduler.step()
            self._steps_executed += 1

            if self._steps_executed % _memory_check_interval == 0:"""
new = """        _burst_steps = int(os.environ.get("VLLM_MLX_DECODE_BURST", "1") or "1")
        _burst_budget_s = float(
            os.environ.get("VLLM_MLX_DECODE_BURST_BUDGET_S", "0.1") or "0.1"
        )

        def _step_on_worker():
            _bind_worker_streams_once()
            output = self.scheduler.step()
            self._steps_executed += 1
            if (
                _burst_steps > 1
                and output.outputs
                and not output.finished_request_ids
            ):
                merged_outputs = list(output.outputs)
                merged_finished = set(output.finished_request_ids)
                merged_sched = list(output.scheduled_request_ids)
                deadline = time.monotonic() + _burst_budget_s
                n = 1
                while (
                    n < _burst_steps
                    and output.outputs
                    and not output.finished_request_ids
                    and self.scheduler.has_requests()
                    and time.monotonic() < deadline
                ):
                    output = self.scheduler.step()
                    self._steps_executed += 1
                    n += 1
                    merged_outputs.extend(output.outputs)
                    merged_finished |= set(output.finished_request_ids)
                    merged_sched.extend(output.scheduled_request_ids)
                output.outputs = merged_outputs
                output.finished_request_ids = merged_finished
                output.scheduled_request_ids = merged_sched

            if self._steps_executed % _memory_check_interval == 0:"""
assert s.count(old) == 1, f"anchor count {s.count(old)}"
open(P, "w").write(s.replace(old, new, 1))
print("burst patch applied")
