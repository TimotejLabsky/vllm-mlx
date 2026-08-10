#!/bin/zsh
# 6-arch vision sweep driver. One model at a time on a spare port, killed
# between runs so peak memory stays at one model. Uses the DEPLOYED
# site-packages (no PYTHONPATH) so this measures the shipped fleet.
set -u
PORT=8096
VENV=/Users/ai/vllm-mlx-env/bin/python
OUT=/tmp/sweep_results.txt
: > "$OUT"

run_one() {
  local arch="$1" repo="$2" seqs="$3"
  echo "" | tee -a "$OUT"
  echo "########## $arch : $repo ##########" | tee -a "$OUT"
  pkill -f "port $PORT" 2>/dev/null
  sleep 4
  nohup env HF_HUB_OFFLINE=1 \
    VLLM_MLX_BATCHED_MEM_WATERMARK_PCT=80 \
    VLLM_MLX_MAX_IMAGES_PER_REQUEST=8 \
    "$VENV" -m vllm_mlx.cli serve "$repo" \
    --host 127.0.0.1 --port $PORT --continuous-batching \
    --max-num-seqs "$seqs" --timeout 600 \
    > "/tmp/sweep_$arch.log" 2>&1 &
  local up=0
  for i in $(seq 1 60); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then up=1; break; fi
    sleep 10
  done
  if [ "$up" -eq 0 ]; then
    echo "  [FAIL] server_up — did not start in 600s" | tee -a "$OUT"
    echo "  --- last log lines ---" | tee -a "$OUT"
    tail -8 "/tmp/sweep_$arch.log" | tee -a "$OUT"
    pkill -f "port $PORT" 2>/dev/null
    return 1
  fi
  "$VENV" /tmp/vision_sweep.py "$PORT" "$arch" 2>&1 | tee -a "$OUT"
  pkill -f "port $PORT" 2>/dev/null
  sleep 4
}

run_one qwen3_5      mlx-community/Qwen3.5-4B-4bit                       8
run_one glm4v        mlx-community/GLM-4.6V-Flash-4bit                   8
run_one mistral3     mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit 4
run_one gemma4       mlx-community/gemma-4-26B-A4B-it-qat-4bit           4
run_one qwen3_vl_moe mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit        4

echo "" | tee -a "$OUT"
echo "########## SWEEP COMPLETE ##########" | tee -a "$OUT"
grep "==>" "$OUT" | tee -a "$OUT"
