#!/usr/bin/env bash
# Score ABot-World's 400 mirror_test predictions with MIND.
#
#   lcm     long-context memory: MSE/LPIPS/SSIM/PSNR vs ground truth after mark_time
#   gsc     general scene consistency: flips the second half of each mirror clip and
#           compares to the first — the loop-closure measurement
#   visual  imaging + aesthetic quality
# Skipped: action (needs ViPE conda env), dino (DINOv3 gated on HF).
set -uo pipefail
VENV=/opt/mindvenv
cd /workspace/repos/MIND
export PYTHONPATH="/workspace/repos/MIND/src:${PYTHONPATH:-}"

GT=/workspace/data/MIND-Data
PRED=${1:-/workspace/preds/ABot-World}
OUT=${2:-/workspace/results/mind_abot.json}
mkdir -p "$(dirname "$OUT")"

echo "[$(date -u +%H:%M:%S)] scoring $PRED"
echo "  clips: $(find "$PRED" -name '*.mp4' | wc -l)"

"$VENV/bin/python" src/process.py \
  --gt_root "$GT" \
  --test_root "$PRED" \
  --num_gpus 1 \
  --metrics lcm,visual,gsc \
  --output "$OUT" 2>&1 | tail -50
echo "[$(date -u +%H:%M:%S)] process.py exit=$?"
ls -la "$OUT" 2>/dev/null
echo "[$(date -u +%H:%M:%S)] EVAL DONE"
