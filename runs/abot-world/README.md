# ABot-World

## Contents

    abot_city_walk.mp4          offline batch rollout, 477 frames / 39.8 s (A100 box)
    sessions/                   live Gradio session recordings (A6000 box)
      stream_20260811_134057.mp4    20 MB — first session
      stream_20260811_230303.mp4   230 MB — 7,197 frames, 12 fps, 10 min continuous
      stream_*.png                  the anchor frame each session started from
    scripts/                    full rebuild: setup, patches, translation, flash-attn
    README.md

The 10-minute session is the interesting artifact — it's long enough to actually
watch the memory degrade, which is the thing worth studying. See the memory notes
in the conversation: `local_attn_size=21`, so anything older than ~21 latent frames
is unreachable, and drift compounds from there.

## Later additions (A6000 rebuild, 2026-08-11)

- **UI translated to English** — 31 strings across 6 files (`scripts/translate_ui.py`),
  verified 0 CJK in rendered text nodes. Originals kept as `*.zh.bak` on the pod.
- **flash-attn 2.8.3 installed** (`scripts/install_fa.sh`). A prebuilt wheel exists
  matching cu12 / torch2.8 / cxx11abiTRUE / cp312 — no source build needed. Note the
  trap: `pip install flash-attn` fails with `Invalid cross-device link` when
  PIP_CACHE_DIR is on a different filesystem; download the wheel and install the file
  directly, keeping its canonical filename or pip rejects it.
  With it installed `FLASH_ATTN_2_AVAILABLE=True`, so `patch_attn.py`'s SDPA fallback
  is bypassed automatically — no need to revert the patch. **Never benchmarked**: the
  pod was terminated before a restart, so the speedup is unmeasured.
- FP8 remains off — that needs Blackwell (sm100+), not more VRAM. The A6000 peaked at
  ~38 GB of 48, so memory was never the constraint.

---

# The original A100 run — Mughal city walk

Same prompt and same anchor image as `Trying/MatrixGame3.5(p)`, for comparison.
Tagged **(p)** — it takes an image, not text.

## What's different about this one

The control signal is an **8-key action sequence**, not a camera trajectory:

    [W, A, S, D, I, J, K, L]     WASD = move, IJKL = look

So this is the only one of the four where the viewpoint is a *player* rather than a
director. The sequence used was a walking tour — mostly forward, panning left at
intervals, one sidestep — cycled over 40 blocks via `--action-mode repeat`.

    abot_city_walk.mp4   477 frames, 12 fps, 39.75 s, 1280x704

Run took ~5.5 min on the A100; peak GPU memory 35.7 GB.

## Three things had to be worked around

This box is an **A100 (sm80)**; ABot targets an **RTX 5090 (Blackwell, sm120)**.

1. **`lightx2v_kernel` missing** — not in requirements.txt, not on PyPI, a CUTLASS source
   build, and its kernels are Blackwell FP4/FP8 which cannot run on sm80 at all.
   Stubbed at `venvs/abot/.../site-packages/lightx2v_kernel/` with a module-level
   `__getattr__` that raises if actually called.
2. **FP8 quantisation disabled** — `use_fp8_gemm: false` in `configs/default_config.yaml`
   (original at `.bak`), and `--quant-type none` on the CLI, which sets
   `use_fp8_gemm = False` at `scripts/inference.py:166`. Runs bf16 instead.
3. **flash-attn 2 hard requirement** — this is the one that actually killed the first run,
   and it is *not* a Blackwell issue. `wan/modules/attention.py:249` asserts
   `FLASH_ATTN_2_AVAILABLE` inside `flash_attention()`, which **cross**-attention calls
   unconditionally at `causal_model.py:1197`. The `ATTN_BACKEND_PRIORITY` list
   (`sla_triton → … → sdpa`) only governs *self*-attention — cross-attention has no
   fallback and just asserts.

   Patched in an SDPA fallback rather than doing a ~30 min flash-attn source build.
   Safe because flash attention is an **exact** algorithm, not an approximation — same
   math, different memory schedule. Verified against a float32 reference: max abs
   difference **0.0039**, i.e. bf16 rounding. Original at `attention.py.bak`.

The first attempt failed with a bare `AssertionError` whose traceback the benchmark
script swallows — it is recoverable from `outputs/profile_bench/*/results.json`, which
stores `traceback` per quant type. Worth knowing if this ever fails again.

## Caveat on speed

ABot advertises 16 FPS at 720p / 19 GB on an RTX 5090 with FP8. None of that applies
here: no FP8, no flash-attn, bf16 on Ampere. Do not read this run's speed as the
model's performance.

## Reproducing

    bash /workspace/abot_run.sh      # pauses the Gradio UI, runs, restarts it

The anchor is not a CLI argument — `BENCH_REF_IMAGE` at `scripts/inference.py:36` is
hardcoded, and the script patches it to `web_client/datasets/images/city_anchor.png`.
