#!/usr/bin/env bash
# Set up three interactive world models on a fresh RunPod A100-80GB.
# Everything lands on /workspace (300 GB volume); nothing touches the 60 GB container disk.
# Separate venv per repo — these three have genuinely conflicting dep pins.
set -uo pipefail

export HF_HOME=/workspace/hf
export PIP_CACHE_DIR=/workspace/pipcache
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBIAN_FRONTEND=noninteractive

mkdir -p /workspace/{hf,pipcache,repos,ckpt,out,logs,venvs}
LOG=/workspace/logs/setup.log
exec > >(tee -a "$LOG") 2>&1

say() { echo -e "\n=== [$(date -u +%H:%M:%S)] $* ==="; }
say "setup start"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h /workspace | tail -1

# ---------------------------------------------------------------- clones (fast)
say "cloning repos"
cd /workspace/repos
clone() {  # clone <url> <dir>
  if [ -d "$2/.git" ]; then echo "  $2 already present"; return 0; fi
  git clone --depth 1 "$1" "$2" && echo "  OK $2" || echo "  FAILED $2"
}
clone https://github.com/Riemann-Dynamics/Matrix-Game-3.5.git Matrix-Game-3.5
clone https://github.com/amap-cvlab/ABot-World.git            ABot-World
clone https://github.com/etched-ai/open-oasis.git             open-oasis
ls -la /workspace/repos

# ------------------------------------------------- weight downloads (background)
# These are the long pole (tens of GB). Kick them off first, install deps while
# they stream. hf_transfer gives us multi-connection downloads.
say "starting weight downloads in background"
# Ubuntu 24.04 marks the system python externally-managed (PEP 668), so a plain
# `python3 -m pip install` here fails. Put the hf CLI in its own venv instead.
python3 -m venv /workspace/venvs/tools
/workspace/venvs/tools/bin/pip install -qU pip huggingface_hub hf_transfer 2>&1 | tail -3
HF=/workspace/venvs/tools/bin/hf
$HF version || { echo "FATAL: hf CLI unavailable"; exit 1; }

dl() {  # dl <repo> <dest> <logname>
  echo "  -> $1"
  $HF download "$1" --local-dir "$2" > "/workspace/logs/dl_$3.log" 2>&1 \
    && echo "  DONE $1" || echo "  FAILED $1 (see logs/dl_$3.log)"
}

dl RiemannDynamics/Matrix-Game-3.5-Distilled /workspace/ckpt/mg35-distilled mg35_distilled &
P1=$!
dl acvlab/ABot-World-0-5B-LF                 /workspace/ckpt/abot-0-5b-lf   abot           &
P2=$!
dl RiemannDynamics/Matrix-Game-3.5-Base      /workspace/ckpt/mg35-base      mg35_base      &
P3=$!

# Open-Oasis weights are gated (gated:auto on Etched/oasis-500m) — needs a token.
if [ -n "${HF_TOKEN:-}" ]; then
  say "HF_TOKEN present — fetching gated Oasis weights"
  $HF download Etched/oasis-500m --local-dir /workspace/ckpt/oasis-500m \
    > /workspace/logs/dl_oasis.log 2>&1 && echo "  DONE oasis-500m" || echo "  FAILED oasis-500m"
else
  echo "  SKIP Etched/oasis-500m — gated, no HF_TOKEN set."
  echo "       Accept terms at https://huggingface.co/Etched/oasis-500m then:"
  echo "       HF_TOKEN=hf_xxx bash /workspace/fetch_oasis.sh"
fi

# ------------------------------------------------------------ per-repo venvs
mkvenv() { python3 -m venv --system-site-packages "/workspace/venvs/$1"; }

say "venv: Matrix-Game-3.5"
mkvenv mg35 && source /workspace/venvs/mg35/bin/activate
python -m pip install -qU pip
[ -f /workspace/repos/Matrix-Game-3.5/requirements.txt ] \
  && python -m pip install -q -r /workspace/repos/Matrix-Game-3.5/requirements.txt 2>&1 | tail -5
python -c "import torch;print('  mg35 torch',torch.__version__,'cuda',torch.cuda.is_available())"
deactivate

say "venv: ABot-World"
mkvenv abot && source /workspace/venvs/abot/bin/activate
python -m pip install -qU pip
[ -f /workspace/repos/ABot-World/requirements.txt ] \
  && python -m pip install -q -r /workspace/repos/ABot-World/requirements.txt 2>&1 | tail -5
python -c "import torch;print('  abot torch',torch.__version__,'cuda',torch.cuda.is_available())"
deactivate

say "venv: open-oasis"
# open-oasis ships no requirements.txt — deps derived from its imports.
mkvenv oasis && source /workspace/venvs/oasis/bin/activate
python -m pip install -qU pip
python -m pip install -q einops timm safetensors tqdm diffusers av numpy \
  rotary-embedding-torch 2>&1 | tail -5
python -c "import torch;print('  oasis torch',torch.__version__,'cuda',torch.cuda.is_available())"
deactivate

# ------------------------------------------------------------ wait on downloads
say "waiting on weight downloads"
wait $P1 $P2 $P3
du -sh /workspace/ckpt/* 2>/dev/null
df -h /workspace | tail -1

# helper for the gated Oasis weights
cat > /workspace/fetch_oasis.sh <<'EOF'
#!/usr/bin/env bash
# Run once you've accepted terms at https://huggingface.co/Etched/oasis-500m
# Usage: HF_TOKEN=hf_xxx bash /workspace/fetch_oasis.sh
set -e
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
hf download Etched/oasis-500m --local-dir /workspace/ckpt/oasis-500m
echo "oasis weights -> /workspace/ckpt/oasis-500m"
EOF
chmod +x /workspace/fetch_oasis.sh

say "setup complete"
echo "repos  -> /workspace/repos/{Matrix-Game-3.5,ABot-World,open-oasis}"
echo "ckpts  -> /workspace/ckpt/"
echo "venvs  -> /workspace/venvs/{mg35,abot,oasis}"
echo "logs   -> /workspace/logs/"
