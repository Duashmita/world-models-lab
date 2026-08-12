# world-models-lab

Notes and trial runs from actually getting open-source interactive world models to run on
rented GPUs — the same text prompt pushed through five different systems, plus the setup
traps that cost real time and money.

This is a lab notebook, not a library. The useful parts are
[`logs/LOG.md`](logs/LOG.md) (every decision, measurement, and dead end) and
[`scripts/`](scripts) (working setups for models whose install instructions don't work
as written).

## The prompt

> A photorealistic bustling mughal city with old Dravida architecture built as tall as
> skyscrappers at dusk in rural india by the lake

Run through everything, so the outputs are directly comparable.
Four prompt variants and why each was needed: [`prompts/prompts.md`](prompts/prompts.md).

## What was run

| model | input | control | output | result |
|---|---|---|---|---|
| **Marble 1.1** (World Labs) | text | — | persistent 3D scene, exportable | hosted |
| **Wan2.2-TI2V-5B** | text | none | 5 s video | `runs/wan2.2/` |
| **Matrix-Game 3.5** | image + camera path | pre-authored trajectory | 31 s directed tour | `runs/matrix-game-3.5/` |
| **Matrix-Game 3.0** | image + actions | terminal REPL | 3.3 s per action | `runs/matrix-game-3.0/` |
| **ABot-World-0.5B** | image + actions | WASD/IJKL, live UI | 10 min session | `runs/abot-world/` |
| Open-Oasis | — | — | — | never ran; gated weights |

Hardware: RunPod A100-80GB and RTX A6000. Total spend across the whole exercise: **~$28**.

## Findings worth the trouble

**"Playable" is not a property most of these have.** Matrix-Game 3.5 has no input loop at
all — you author a full camera trajectory, then it renders. Matrix-Game 3.0 is a terminal
REPL. ABot is the only one shipping a web UI. Under World Labs' own taxonomy, Marble is a
*Renderer* and the rest are *Simulators*; that single distinction explains most of the
behavioural differences.

**None of the video models keep a scene.** ABot's self-attention runs a 21-latent-frame
window (`local_attn_size=21`) with no depth model and no 3D cache — walk away and come
back, and the geometry was never stored to begin with. Matrix-Game 3.5 at least maintains
an explicit 3D cache ("Mosaic Memory") plus Depth-Anything-3, and still drifts. This is
the field's open problem, not a configuration mistake: see MBench, LoopNav, WorldRoamBench.

**Defaults will quietly waste your money.** `configs/infer_distilled.yaml` ships
`num_blocks: 1`, capping output at 85 frames from a 961-frame camera path — 8.8% of the
trajectory. Six clips were rendered before this was caught; they look static because they
*are* static. The 6-block config sits in the same directory.

**Measured, not guessed:** ABot generates 85.4 frames/min on an A100 (bf16, no FP8).
HF downloads sustain ~380 MB/s. Matrix-Game 3.5 peaks at 67 GB VRAM; ABot idles at 22 GB
with its model resident — so one A100 cannot host both, and they will OOM together.

## Traps

Full list in [`logs/LOG.md`](logs/LOG.md). The ones most likely to bite anyone else:

- **ABot needs three separate workarounds on Ampere.** Its `lightx2v_kernel` dependency is
  Blackwell-only FP4/FP8 CUTLASS (stub it, set `use_fp8_gemm: false`), and — the one that
  actually blocks rollouts — `wan/modules/attention.py` asserts `FLASH_ATTN_2_AVAILABLE`
  inside a function cross-attention calls unconditionally. `ATTN_BACKEND_PRIORITY` only
  governs *self*-attention. That one is not a Blackwell issue; flash-attn 2 works fine on
  Ampere, it just wasn't installed. `scripts/abot/patch_attn.py` adds an exact SDPA fallback.
- **flash-attn has a prebuilt wheel** for cu12/torch2.8/cxx11abiTRUE/cp312 — no source
  build. But `pip install flash-attn` fails with `Invalid cross-device link` if
  `PIP_CACHE_DIR` is on another filesystem. Curl the wheel, install the file, keep its
  canonical name.
- **Matrix-Game 3.0's `requirements.txt` pins `torch==2.10.0`.** Against any other torch
  this is `ResolutionImpossible` and the install fails silently. Strip the torch lines.
- **Matrix-Game 3.0's REPL hangs forever on stdin EOF** — `get_current_action()` wraps
  `input()` in a bare `except: pass`. It needs a live TTY; never pipe into it.
- **Matrix-Game 3.5's `requirements.txt` omits `ftfy` and `av`**, both needed at runtime,
  and it needs two extra base models not in its checkpoint (Wan2.2-TI2V-5B, 32 GB, and
  Depth-Anything-3, 6.4 GB).
- **pip resolver backtracking** on unpinned `torch` + `xformers`: 10+ minutes with open
  sockets and zero writes to site-packages. That's the signature. A constraints file
  pinning torch turns the same install into 30 seconds.
- **RunPod:** passing an `env` dict to pod creation replaces the whole env set, wiping the
  auto-injected `PUBLIC_KEY` so sshd never starts. The public SSH port remaps on every
  restart. Pods are terminated outright when the account balance hits zero.

## Layout

```
logs/LOG.md          decisions, measurements, cost arithmetic, traps, corrections
prompts/prompts.md   four prompt variants and the reasoning
scripts/             working setup + patches per model
runs/                outputs, per model
```

`scripts/matrix-game-3.5/make_camera.py` generates camera trajectories (dolly, truck,
pedestal, pan, tilt, orbit, crane) in the `.npz` format Matrix-Game expects — the
convention is documented from reading the bundled samples, since it isn't written down
anywhere.

## Caveats

Every model here ran **degraded**: Ampere GPUs, no FP8, and in ABot's case a stubbed
quantisation path. ABot advertises 16 FPS on an RTX 5090; nothing here should be read as
a performance benchmark. Quality comparisons are fair (same prompt, same anchor);
speed comparisons are not.

Nothing in `runs/` is cherry-picked — the failed and static clips are included alongside
the good ones, because the failures are what the log is about.
