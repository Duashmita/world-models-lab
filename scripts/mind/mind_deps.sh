#!/usr/bin/env bash
# MIND eval venv on the CONTAINER DISK, not /workspace.
# /workspace is a network filesystem (mfs) and threw [Errno 5] Input/output error partway
# through a ~50-package install. / is local, 60 GB free, and far faster for many small files.
# The venv is ephemeral, which is fine — only the results JSON needs to persist.
export PIP_CACHE_DIR=/root/pipcache
mkdir -p /root/pipcache
VENV=/opt/mindvenv
rm -rf "$VENV"
python3 -m venv --system-site-packages "$VENV"
PIP="$VENV/bin/pip"
echo "[$(date -u +%H:%M:%S)] installing MIND eval deps to $VENV"
$PIP install -qU pip 2>&1 | tail -1
$PIP install --constraint /workspace/constraints.txt \
  -r /workspace/repos/MIND/envs/requirements.txt 2>&1 | tail -6
echo "[$(date -u +%H:%M:%S)] verify"
cd /workspace/repos/MIND
PYTHONPATH=/workspace/repos/MIND/src "$VENV/bin/python" -c "import torch, lpips, torchmetrics, pyiqa, clip; print('  imports OK | torch', torch.__version__, '| lpips', lpips.__version__ if hasattr(lpips,'__version__') else 'ok')"
echo "[$(date -u +%H:%M:%S)] MIND DEPS DONE"
