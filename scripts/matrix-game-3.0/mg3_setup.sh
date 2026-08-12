#!/usr/bin/env bash
# Matrix-Game 3.0 — interactive terminal REPL setup.
# MG-3.0 is Skywork's real-time streaming model; --interactive reads mouse/keyboard
# actions from stdin and generates 40-57 frames per action.
set -uo pipefail
export HF_HOME=/workspace/hf PIP_CACHE_DIR=/workspace/pipcache DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/{hf,pipcache,repos,ckpt,logs,venvs}
LOG=/workspace/logs/setup.log
exec > >(tee -a "$LOG") 2>&1
say(){ echo -e "\n=== [$(date -u +%H:%M:%S)] $* ==="; }

say "box"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

say "clone Matrix-Game"
cd /workspace/repos
[ -d Matrix-Game/.git ] || git clone --depth 1 https://github.com/SkyworkAI/Matrix-Game.git
ls Matrix-Game/Matrix-Game-3

say "tools venv + checkpoint download (~57 GB) -> background"
python3 -m venv /workspace/venvs/tools
/workspace/venvs/tools/bin/pip install -qU pip huggingface_hub 2>&1 | tail -2
setsid nohup /workspace/venvs/tools/bin/hf download Skywork/Matrix-Game-3.0 \
  --local-dir /workspace/ckpt/mg3 > /workspace/logs/dl_mg3.log 2>&1 < /dev/null &
DL=$!

say "venv + deps"
python3 -m venv --system-site-packages /workspace/venvs/mg3
PIP=/workspace/venvs/mg3/bin/pip
$PIP install -qU pip 2>&1 | tail -1
# Pin what the image already provides so the resolver cannot backtrack over torch.
cat > /workspace/constraints.txt <<'CEOF'
torch==2.8.0+cu128
torchvision==0.23.0+cu128
CEOF
cd /workspace/repos/Matrix-Game/Matrix-Game-3
echo "--- requirements.txt ---"; cat requirements.txt
$PIP install --constraint /workspace/constraints.txt -r requirements.txt 2>&1 | tail -6
$PIP check || true

say "flash-attn 2 (README wants fa_version 3 = Hopper; this box is Ampere, so fa2)"
W="flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
curl -sSL -o "/workspace/$W" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/$W" \
  && $PIP install -q --no-deps --no-cache-dir "/workspace/$W" 2>&1 | tail -2
/workspace/venvs/mg3/bin/python -c "import flash_attn;print('  flash_attn',flash_attn.__version__)" 2>&1|tail -1

say "torch sanity (must still be the image's build)"
/workspace/venvs/mg3/bin/python -c "import torch;print(' ',torch.__version__,'cuda',torch.cuda.is_available())"

say "waiting on checkpoint"
wait $DL 2>/dev/null
du -sh /workspace/ckpt/mg3; ls /workspace/ckpt/mg3 | head

say "setup complete"
