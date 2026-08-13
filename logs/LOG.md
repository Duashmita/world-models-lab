# World models — working log

Decisions, measurements, prompts, and dead ends. Appended as we go.
Newest section at the bottom.

---

## Cast of models

| model | input | control | output | ran it? |
|---|---|---|---|---|
| Marble 1.1 (World Labs) | text | — | persistent 3D scene (exportable) | yes, by you |
| Wan2.2-TI2V-5B | text (image optional) | none | 5 s video | yes |
| Matrix-Game 3.5 | image + camera path | camera trajectory | video, batch only | yes |
| Matrix-Game 3.0 | image + actions | keyboard/mouse via stdin REPL | video, real-time claims | in progress |
| ABot-World-0-5B-LF | image + actions | 8-key WASD/IJKL | video, live Gradio UI | yes |
| Open-Oasis | image | — | video | **blocked** — gated weights |

Marble is a **Renderer** in World Labs' taxonomy; the rest are **Simulators**.
That's why Marble persists and the others drift.

---

## Key decisions and why

**Chose Matrix-Game 3.5 over Oasis-500m.** Oasis is Oct-2024, Minecraft-only, and its
weights are gated. MG-3.5 is Apache-2.0, ungated, 720p. The gating cost a whole pod
($15.04) for zero output on the first attempt.

**Chose A100-80GB over A6000 for MG-3.5.** MG-3.5 needs ≥40 GB VRAM; measured peak was
67 GB during causal KV rollout.

**Chose A6000 ($0.53/h) over A100 ($1.59/h) for ABot-only.** ABot peaked at 35.7 GB of
48 — never memory-bound. "Bigger GPU" was the wrong axis; the bottlenecks were FP8
(needs Blackwell sm100+) and flash-attn, both architectural.

**Skipped the Blackwell option** despite it being ABot's actual target (RTX PRO 6000,
$2.09/h, sm120, real FP8). It needs a `lightx2v_kernel` CUTLASS build, ~30 min and not
guaranteed — not worth stacking onto an already multi-step task.

**Skipped ViPE + DINOv3 for MIND.** Dependency audit of `src/metrics/`:
`lcm` needs torch/lpips/torchmetrics, `visual` needs MUSIQ+aesthetic, `gsc` reuses the
lcm machinery — **none need ViPE**. Only `action` (via `utils.vipe_utils`) and `dino`
do. Since lcm and gsc *are* the memory metrics, we skip the conda/ViPE install entirely.
DINOv3 is gated anyway ("Access denied. This repository requires approval").

**Scoped the MIND run down** from the full benchmark. See calculations below.

---

## Measurements (all measured, not estimated)

| thing | value | how |
|---|---|---|
| HF download throughput | ~380 MB/s | 54.8 GB in ~3 min, A100 box |
| ABot generation rate | **85.4 frames/min** | 477 frames in 5 m 35 s, A100, SDPA, no FP8 |
| ABot idle VRAM (model resident) | ~22 GB | blocked MG-3.5 → OOM |
| ABot peak VRAM (bf16, no FP8) | 35.7 GB | offline batch run |
| MG-3.5 peak VRAM | 67 GB | causal KV rollout, 6-block |
| MG-3.5 distilled, 1 block | 85 frames, ~3 min | measured |
| MG-3.5 distilled, 6 blocks | 505 frames, 13.5 min | 42 chunks |
| Motion, 1-block clip | 16.5 / 255 | first-vs-last frame mean abs diff |
| Motion, 6-block tour | 61.4 / 255 | ~3.7× more movement |
| Wan2.2 T2V motion / sec | 9.8 / 255 | 1 s apart |
| MG-3.5 tour motion / sec | 11.7 / 255 | 1 s apart |

**Note on that last pair:** I predicted Wan would animate the scene more than MG-3.5.
It measured *lower*. Caveat: mean pixel diff cannot separate camera motion from object
motion, so MG-3.5's number is inflated by camera translation.

---

## Cost calculations

### MIND full benchmark (rejected)

The dataset has 250 `action.json`, but **100 are a train split** — the test set is 150.
(I initially said 243; that was wrong, it included train.)

| split | samples | frames to predict |
|---|---|---|
| 1st_data / mem_test | 50 | 113,896 |
| 1st_data / action_space_test | 25 | 62,801 |
| 3rd_data / mem_test | 50 | 112,225 |
| 3rd_data / action_space_test | 25 | 61,461 |
| **subtotal** | **150** | **350,383** |
| mirror_test (50 anchors × 10 paths × 2 persp × 95 f) | 1,000 clips | 95,000 |
| **total** | | **~445,000** |

At the measured 85.4 frames/min:
- 350,383 f → 68.4 h → **2.85 days → $109**
- +mirror_test → ~87 h → **~$138**
- with flash-attn at 1.5× → $72 · at 2× → $54

Sample sizes: median `total_time` 3,341, median `mark_time` 1,113 → condition ~46 s,
predict ~93 s at 24 fps. Max 7,358 frames.

### Scoped run (chosen)

mirror_test subset — 10 anchors × 10 paths × 2 perspectives = 200 clips × 95 frames
= **19,000 frames ≈ 3.7 h ≈ $6**, plus a handful of `mem_test` for the long-horizon number.

Rationale: mirror_test clips are short (95 f) *and* directly measure loop closure,
which is the failure actually observed. Best signal per dollar.

### Actual spend so far

| pod | what | cost |
|---|---|---|
| `xz036vm8dvpx2i` | first A100, never cleared HF gating | $15.04 |
| `m24fyk9980huz7` | MG-3.5 + ABot + Wan2.2 work | $10.02 |
| `fh9oj3aq7q3tih` | A6000, ABot live UI | ~$0.30 |
| | **before the MIND/MG3 pair** | **~$25.4** |

---

## How MIND actually scores memory

From `src/process.py`:

```python
mirror_pred = torch.flip(imgs[sample_frames // 2:], dims=[0])
gsc = lcm_metric(origin_pred, mirror_pred, ...)
```

So each `path-N.mp4` must contain **outbound + return concatenated**, with `mark_time`
as the divider. MIND flips the second half and compares it against the first — if the
model truly remembered the scene, walking back should retrace what it drew going out.
That is the loop-closure metric, and it's elegant.

`lcm` = long-context memory (MSE / LPIPS / SSIM / PSNR vs ground truth after `mark_time`).

### Action-space mapping

MIND `action.json` per frame:
```json
{"time":0, "ws":0, "ad":0, "ud":0, "lr":0,
 "actor_pos":{x,y,z}, "actor_rpy":{x,y,z}}
```
Maps cleanly onto ABot's `[W,A,S,D,I,J,K,L]`:
`ws→W/S`, `ad→A/D`, `ud→I/K`, `lr→J/L`. Ground-truth Unreal poses and COLMAP
`images.txt` are included for the (skipped) action metric.

The 10 mirror paths: `-w, -a, -s, -d, -wl, -wr, -sl, -sr, -u, -down`.

---

## Prompts used

Full text in `Trying/prompts.md`. Four versions:

- **v0** — yours, verbatim, used on Marble
- **v1** — expanded static description → the six 1-block MG-3.5 clips + the anchor image
- **v2** — motion-heavy (boats rocking, lanterns pulsing, wind) → the 6-block tour,
  Wan2.2 T2V, and the ABot batch run
- **v3** — walkable/first-person, drops "the camera glides forward" because in ABot
  *you* are the camera; adds ground texture and navigable-path cues, mirroring ABot's
  own default prompt style

**Why expand at all:** MG-3.5's bundled sample prompts are full paragraphs specifying
material, light direction, and motion. Short prompts underperform. Its config also
carries a negative prompt stuffed with 静止 / 静止不动的画面 ("static", "motionless
picture") — the authors are actively fighting stasis, which says it's a known failure mode.

---

## Bugs, traps, and fixes

**`create-pod` with an `env` dict replaces the whole env set**, wiping RunPod's
auto-injected `PUBLIC_KEY` → sshd never starts, SSH refused with no obvious cause.
Pass `PUBLIC_KEY` explicitly.

**Pod restart remaps the public SSH port** every time (13009→10795, 46090→43321).
Re-read from `get-pod → runtime.ports`.

**Ubuntu 24.04 system python is PEP 668 externally-managed** — plain `pip install`
hard-fails. Everything needs a venv.

**pip resolver backtracking**: unpinned `torch` + `xformers` ran 10+ minutes with open
sockets and *zero* writes to site-packages. That's the tell — sockets busy, disk idle,
low CPU. Pin torch/torchvision via `--constraint` and the same install finishes in ~30 s.

**ABot's pins only install via `--no-deps`** then a `pip check` gap-fill loop. Full
resolution never converges. Watch for over-new resolutions: `antlr4-runtime` breaks
`omegaconf`, `tomlkit` and `aiofiles` break `gradio`.

**A killed `hf download` leaves `.lock` files that deadlock every retry**, plus an
orphaned `.incomplete` at full byte count showing 0 B/s. Clear both.

**`pkill -f <pattern>` over SSH matches your own command line** and kills the session
(exit 255). Use `[p]attern`. Note the pattern also appears inside heredocs.

**zsh does not word-split unquoted `$VAR`** — `ssh $OPTS host` passes one mangled arg.
Write ssh flags explicitly or use `${=VAR}`.

**MG-3.5 `requirements.txt` is incomplete** — `ftfy` and `av` needed at runtime, neither
listed. Its README's `--exclude "a" "b"` is also wrong for hf ≥1.26 (second pattern
becomes a positional filename); use two `--exclude` flags.

**MG-3.5 needs two extra base models** not in its checkpoint: `Wan2.2-TI2V-5B` (32 GB)
and `DA3NESTED-GIANT-LARGE-1.1` (6.4 GB).

**`num_blocks: 1` is the default** in `configs/infer_distilled.yaml` — caps output at 85
frames from a 961-frame camera path, i.e. 8.8% of the trajectory. This is why the first
six city clips looked static. `configs/infer_distilled_6blocks.yaml` ships alongside it.

**Camera trajectory scale matters**: bundled samples move 0.002–0.03 units/frame. My
first hand-authored tour was 0.0806 — ~2.7× too fast, would tear. Rescaled to 0.0329.

**ABot on Ampere needs three workarounds**: stub `lightx2v_kernel` (Blackwell FP4/FP8,
sm100+), `--quant-type none` / `use_fp8_gemm:false`, and — the one that actually blocks
rollouts — an SDPA fallback in `wan/modules/attention.py`. `flash_attention()` asserts
`FLASH_ATTN_2_AVAILABLE` and cross-attention calls it unconditionally;
`ATTN_BACKEND_PRIORITY` only covers *self*-attention. **That one is not a Blackwell
issue** — flash-attn 2 supports Ampere, it just wasn't installed.

**flash-attn has a prebuilt wheel** for cu12/torch2.8/cxx11abiTRUE/cp312 — no source
build. But `pip install flash-attn` fails with `Invalid cross-device link` when
`PIP_CACHE_DIR` is on a different filesystem; curl the wheel and install the file,
keeping its canonical name or pip rejects it.

**ABot's benchmark script swallows tracebacks** — recover from
`outputs/profile_bench/*/results.json`, which stores `traceback` per quant type.

**One A100 cannot host ABot + MG-3.5 simultaneously.** ABot idles at 22 GB; MG-3.5 peaks
at 67 GB. They OOM'd together. Sequence them.

**RunPod terminates running pods when the account balance hits zero.** That, not any
command, killed the first two pods. `create-pod` then returns 402.

---

## Stop vs terminate — a wrong call, corrected

I advised **stopping** the pods rather than terminating, on the reasoning that
`/workspace` persists and resume takes ~2 minutes. That's wrong in a way that matters:

> **A stopped RunPod pod does not reserve its GPU.**

Both hosts filled while the pods were stopped, and `start-pod` requires the *original*
machine. Neither could restart:

    400 — "There are not enough free GPUs on the host machine to start this pod."

A100s showed HIGH availability for *new* pods the whole time — it's host-specific, and
there's no way to migrate a pod volume to a different host. Had to terminate and
recreate, losing 112 GB of downloaded weights and datasets.

**Rule:** for a break of hours or more, terminate and rebuild from scripts. Stop only
wins for short breaks where you'll return to the same machine. Cost of being wrong is
the rebuild; cost of being right is a few cents of volume storage — but the rebuild only
stays cheap if the setup is fully scripted, which is the real argument for scripting it.

Rebuild took **7 minutes** for both pods, versus ~40 minutes the first time, because by
then the scripts carried every fix.

---

## RESUME STATE (2026-08-12, after rebuild)

Pod IDs below are the **second** generation; the originals were terminated (see above).
SSH port remaps on every start — re-read from `get-pod → runtime.ports`.

**`lbewwyyec5ntys` "mind-abot-2"** — A100-80GB, US-MD-1, $1.59/h, 250 GB vol
- `/workspace/data/MIND-Data` — 35 GB, all 730 files
- `/workspace/ckpt/abot` — 24 GB; `/workspace/repos/{MIND,ABot-World}`
- `/workspace/venvs/abot` — deps clean, flash-attn 2.8.3, all three ABot patches applied
- **Next step:** write `action.json` → ABot `[W,A,S,D,I,J,K,L]` bridge, generate the
  mirror_test subset, then `python src/process.py --gt_root /workspace/data/MIND-Data
  --test_root <preds> --num_gpus 1 --metrics lcm,visual,gsc`
- Skip ViPE and DINOv3 (only `action`/`dino` need them; DINOv3 is gated anyway)

**`ksyndrsqfzdoax` "matrix3-play-2"** — A100-80GB, US-KS-2, $1.59/h, 200 GB vol
- `/workspace/ckpt/mg3` — 53 GB; `/workspace/repos/Matrix-Game`
- `/workspace/play.sh` — interactive REPL, verified working
- Play: `ssh -t root@<ip> -p <port> -i ~/.ssh/id_ed25519 'bash /workspace/play.sh'`
- Controls: camera `I/K/J/L/U`, movement `W/S/A/D/Q`. 57 frames @ 17 fps per action.

**MG-3.0 gotchas:** `requirements.txt` pins torch==2.10.0 → ResolutionImpossible against
the image's 2.8.0; strip the torch/torchvision/torchaudio lines. `torchrun` isn't in the
venv (torch is system-installed). `--fa_version 3` is Hopper-only → use 2 on Ampere.
**Its REPL hangs forever on stdin EOF** — `get_current_action()` wraps `input()` in a
bare `except: pass`. Needs a live TTY; never pipe input.

Everything needed to rebuild either box from scratch is in
`Trying/*/scripts/`. Nothing depends on the pods.

---

## Corrections I had to make

Recorded because they're the useful part.

1. Claimed MG-3.5 would be blocked on gated weights like Oasis — it isn't, it's ungated.
2. Predicted Wan2.2 would show more scene motion than MG-3.5. Measured the opposite.
3. Said this pod had cost ~$6.60 when billing showed **$10.02** — I was estimating from
   wall-clock instead of reading the billing API, while you were making stop/continue
   calls on those numbers.
4. Blamed ABot's first offline failure on Blackwell/sm80. It was a missing flash-attn 2,
   which works fine on Ampere.
5. Said the MIND benchmark was 243 samples / ~4.7 days. It's **150 test samples**
   (100 of the 250 are train) / **2.85 days**.
6. Left the ABot UI running during the first MG-3.5 tour, causing an OOM. My error, not
   a model limitation.
