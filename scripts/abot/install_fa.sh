#!/usr/bin/env bash
# Install flash-attn 2 WITHOUT disturbing the running UI.
# Only takes effect on the next UI restart (FLASH_ATTN_2_AVAILABLE is set at import).
# Capped at 32 of 128 cores and niced so the live rollout keeps its CPU.
export PIP_CACHE_DIR=/workspace/pipcache
export MAX_JOBS=32
PIP=/workspace/venvs/abot/bin/pip
echo "[$(date -u +%H:%M:%S)] trying prebuilt wheel first"
nice -n 15 $PIP install flash-attn --no-build-isolation 2>&1 | tail -25
echo "[$(date -u +%H:%M:%S)] exit=$?"
echo "=== installed? ==="
$PIP list 2>/dev/null | grep -i flash || echo "  NOT installed"
