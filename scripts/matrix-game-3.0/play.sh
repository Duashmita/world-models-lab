#!/usr/bin/env bash
# Matrix-Game 3.0 — interactive terminal REPL.
#   CAMERA:   I up · K down · J left · L right · U none
#   MOVEMENT: W fwd · S back · A left · D right · Q none
# It asks for camera first, then movement, and generates ~40-57 frames per action.
# Adapted for 1x A100 (Ampere): their test.sh assumes 8 GPUs and --fa_version 3 (Hopper).
cd /workspace/repos/Matrix-Game/Matrix-Game-3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IMG=${IMG:-demo_images/001/image.png}
PROMPT=${PROMPT:-"A colorful, animated cityscape with a gas station and various buildings."}
exec /workspace/venvs/mg3/bin/python -m torch.distributed.run --nproc_per_node=1 generate.py \
  --size 704*1280 \
  --ckpt_dir /workspace/ckpt/mg3 \
  --output_dir /workspace/output \
  --fa_version 2 \
  --num_iterations 12 \
  --num_inference_steps 3 \
  --image "$IMG" \
  --prompt "$PROMPT" \
  --save_name play --seed 42 \
  --vae_type mg_lightvae \
  --interactive
