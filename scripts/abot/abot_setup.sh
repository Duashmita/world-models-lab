#!/usr/bin/env bash
# ABot-World only, with every workaround from the previous build applied up front.
set -uo pipefail
export HF_HOME=/workspace/hf PIP_CACHE_DIR=/workspace/pipcache DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/{hf,pipcache,repos,ckpt,logs,venvs}
LOG=/workspace/logs/setup.log
exec > >(tee -a "$LOG") 2>&1
say(){ echo -e "\n=== [$(date -u +%H:%M:%S)] $* ==="; }

say "box"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

say "clone"
cd /workspace/repos
[ -d ABot-World/.git ] || git clone --depth 1 https://github.com/amap-cvlab/ABot-World.git ABot-World

# hf CLI in its own venv (Ubuntu 24.04 system python is PEP 668 externally-managed)
say "tools venv + weight download (24 GB)"
python3 -m venv /workspace/venvs/tools
/workspace/venvs/tools/bin/pip install -qU pip huggingface_hub 2>&1 | tail -2
setsid nohup /workspace/venvs/tools/bin/hf download acvlab/ABot-World-0-5B-LF \
  --local-dir /workspace/ckpt/abot > /workspace/logs/dl.log 2>&1 < /dev/null &
DL=$!

say "abot venv (--no-deps: its pins never resolve otherwise)"
python3 -m venv --system-site-packages /workspace/venvs/abot
PIP=/workspace/venvs/abot/bin/pip
$PIP install -qU pip 2>&1 | tail -2
$PIP install --no-deps -r /workspace/repos/ABot-World/requirements.txt 2>&1 | tail -3
say "gap-fill transitive deps"
$PIP install -q huggingface-hub importlib-metadata "aiofiles<25.0" fastapi pandas \
  safehttpx typer uvicorn contourpy annotated-types scipy tifffile 2>&1 | tail -3
$PIP check || true

say "patch 1/3 — lightx2v_kernel stub (Blackwell FP4/FP8 kernels, unrunnable on Ampere)"
SP=$(/workspace/venvs/abot/bin/python -c "import site;print(site.getsitepackages()[0])")
mkdir -p "$SP/lightx2v_kernel"
cat > "$SP/lightx2v_kernel/__init__.py" <<'EOF'
"""Stub: real package is CUTLASS FP4/FP8 GEMM requiring sm100+. Requires use_fp8_gemm:false."""
EOF
cat > "$SP/lightx2v_kernel/gemm.py" <<'EOF'
def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"lightx2v_kernel.gemm.{name} needs Blackwell sm100+ kernels; "
                           "this GPU is Ampere. Keep use_fp8_gemm: false.")
    _f.__name__ = name
    return _f
EOF

say "patch 2/3 — disable FP8, repoint Gradio to the exposed port"
cd /workspace/repos/ABot-World
cp configs/default_config.yaml configs/default_config.yaml.bak
sed -i 's/^use_fp8_gemm: true/use_fp8_gemm: false/' configs/default_config.yaml
sed -i 's/^SERVER_PORT = .*/SERVER_PORT = 7860/' web_client/config.py
grep -E '^use_fp8_gemm' configs/default_config.yaml
grep -E '^SERVER_PORT|^SERVER_NAME' web_client/config.py

say "patch 3/3 — SDPA fallback for cross-attention (flash-attn 2 absent)"
cp wan/modules/attention.py wan/modules/attention.py.bak
/workspace/venvs/abot/bin/python /workspace/patch_attn.py
/workspace/venvs/abot/bin/python -c "import ast;ast.parse(open('wan/modules/attention.py').read());print('  syntax OK')"

say "wait for weights"
wait $DL
du -sh /workspace/ckpt/abot
mkdir -p checkpoints
ln -sfn /workspace/ckpt/abot checkpoints/ABot-World-0-5B-LF
ls -la checkpoints/

say "setup complete"
