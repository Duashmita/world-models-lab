#!/usr/bin/env bash
# Stage 1: things we already know how to do — dataset + ABot rebuild.
# ViPE/conda (stage 2) is the risky part and runs separately.
set -uo pipefail
export HF_HOME=/workspace/hf PIP_CACHE_DIR=/workspace/pipcache DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/{hf,pipcache,repos,ckpt,logs,venvs,data}
LOG=/workspace/logs/stage1.log
exec > >(tee -a "$LOG") 2>&1
say(){ echo -e "\n=== [$(date -u +%H:%M:%S)] $* ==="; }

say "tools venv (hf CLI)"
python3 -m venv /workspace/venvs/tools
/workspace/venvs/tools/bin/pip install -qU pip huggingface_hub 2>&1 | tail -2
HF=/workspace/venvs/tools/bin/hf

say "MIND dataset (ground truth) -> background"
setsid nohup $HF download CSU-JPG/MIND --repo-type dataset \
  --local-dir /workspace/data/MIND-Data > /workspace/logs/dl_mind.log 2>&1 < /dev/null &
DL1=$!

say "DINOv3 weights -> background"
setsid nohup $HF download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir /workspace/ckpt/dinov3_vitb16 > /workspace/logs/dl_dino.log 2>&1 < /dev/null &
DL2=$!

say "ABot weights -> background"
setsid nohup $HF download acvlab/ABot-World-0-5B-LF \
  --local-dir /workspace/ckpt/abot > /workspace/logs/dl_abot.log 2>&1 < /dev/null &
DL3=$!

say "clone ABot"
cd /workspace/repos
[ -d ABot-World/.git ] || git clone --depth 1 https://github.com/amap-cvlab/ABot-World.git ABot-World

say "abot venv (--no-deps, then gap-fill — its pins never resolve otherwise)"
python3 -m venv --system-site-packages /workspace/venvs/abot
PIP=/workspace/venvs/abot/bin/pip
$PIP install -qU pip 2>&1 | tail -1
$PIP install --no-deps -r /workspace/repos/ABot-World/requirements.txt 2>&1 | tail -3
for round in 1 2 3; do
  MISSING=$($PIP check 2>/dev/null | grep -oP "requires \K[a-zA-Z0-9_.-]+(?=, which is not installed)" | sort -u | tr "\n" " ")
  [ -z "$MISSING" ] && break
  echo "  gap-fill round $round: $MISSING"
  $PIP install -q $MISSING 2>&1 | tail -1
done
# known-bad resolutions from last time
$PIP install -q "aiofiles<25.0" "antlr4-python3-runtime==4.9.3" "tomlkit>=0.12.0,<0.14.0" 2>&1 | tail -1
$PIP check || true

say "flash-attn 2 (prebuilt wheel — matches cu12/torch2.8/cxx11abiTRUE/cp312)"
W="flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
curl -sSL -o "/workspace/$W" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/$W" \
  && $PIP install -q --no-deps --no-cache-dir "/workspace/$W" 2>&1 | tail -2
/workspace/venvs/abot/bin/python -c "import flash_attn;print('  flash_attn',flash_attn.__version__)" 2>&1|tail -1

say "ABot patches"
cd /workspace/repos/ABot-World
SP=$(/workspace/venvs/abot/bin/python -c "import site;print(site.getsitepackages()[0])")
mkdir -p "$SP/lightx2v_kernel"
echo '"""Stub: real pkg is CUTLASS FP4/FP8, needs sm100+."""' > "$SP/lightx2v_kernel/__init__.py"
cat > "$SP/lightx2v_kernel/gemm.py" <<'EOF'
def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"lightx2v_kernel.gemm.{name} needs Blackwell sm100+; keep use_fp8_gemm: false.")
    _f.__name__ = name
    return _f
EOF
cp configs/default_config.yaml configs/default_config.yaml.bak
sed -i 's/^use_fp8_gemm: true/use_fp8_gemm: false/' configs/default_config.yaml
# SDPA fallback is only needed when flash-attn is absent; apply anyway for safety
cp wan/modules/attention.py wan/modules/attention.py.bak
/workspace/venvs/abot/bin/python /workspace/patch_attn.py || echo "  (patch skipped/already applied)"
grep -E '^use_fp8_gemm' configs/default_config.yaml

say "waiting on downloads"
wait $DL1 $DL2 $DL3 2>/dev/null
du -sh /workspace/data/MIND-Data /workspace/ckpt/* 2>/dev/null
mkdir -p checkpoints && ln -sfn /workspace/ckpt/abot checkpoints/ABot-World-0-5B-LF

say "stage 1 complete"
