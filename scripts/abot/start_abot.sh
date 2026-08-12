#!/usr/bin/env bash
export HF_HOME=/workspace/hf CUDA_VISIBLE_DEVICES=0 ABOTWORLD_DEBUG_FRONTEND=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/repos/ABot-World
exec /workspace/venvs/abot/bin/python web_client/app.py
