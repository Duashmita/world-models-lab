# MIND × ABot-World — loop-closure benchmark

Running [MIND](https://github.com/CSU-JPG/MIND) (arXiv [2602.08025](https://arxiv.org/abs/2602.08025))
against [ABot-World-0.5B](https://github.com/amap-cvlab/ABot-World) to put a number on the
memory failure that's obvious by eye: walk somewhere, turn around, and the model has
invented a different world.

ABot is the model under test. MIND is the measuring instrument.

## What `gsc` actually measures

MIND's mirror_test paths are **out-and-back by construction**. From `src/process.py`:

```python
mirror_pred = torch.flip(imgs[sample_frames // 2:], dims=[0])
gsc = lcm_metric(origin_pred, mirror_pred, ...)
```

It flips the return half and compares it to the outbound half. A model that remembered
the scene would retrace what it drew going out, so the two halves would be near-identical:
**SSIM → 1, LPIPS → 0, PSNR 30 dB+**.

The action files make the design explicit — every path undoes itself:

```
-w    W W  →  S S                    forward, then reverse
-a    A A  →  D D                    left, then right
-wl   W W  →  J J  →  L L  →  S S    forward, look left, look right, back
```

## Result

**400 clips generated** (20 anchors × 10 paths × 2 perspectives, mirror_test split).
**329 scored** — the RunPod account balance ran out and the pod was stopped mid-evaluation,
so the last 7 anchors were never scored and the final JSON is stranded on the stopped
volume. The numbers below are n=329 videos across 33 of 40 anchors.

| metric | ABot-World | what "remembered" looks like |
|---|---|---|
| SSIM ↑ | **0.257** | ~1.0 |
| LPIPS ↓ | **0.641** | ~0.0 |
| PSNR ↑ | **10.64 dB** | 30 dB+ |
| MSE ↓ | 0.090 | ~0.0 |

The truncation doesn't affect the conclusion. The mean was stable to four significant
figures across every partial — 70, 199, 209 and 329 videos:

| n videos | SSIM | LPIPS | PSNR |
|---|---|---|---|
| 70 | 0.2568 | 0.6422 | 10.615 |
| 199 | 0.2568 | 0.6414 | 10.638 |
| 209 | 0.2572 | 0.6415 | 10.643 |
| 329 | 0.2571 | 0.6414 | 10.644 |

and first- vs third-person agree to three decimals (SSIM 0.2568 vs 0.2576, MSE 0.0897 for
both). The estimate had converged long before it stopped.

At ~10 dB PSNR the returned view is closer to unrelated imagery than to where it started.
ABot isn't *misremembering* the scene — it's regenerating a new one.

**The failure is uniform, not occasional.** SSIM ranges 0.176–0.312 across all clips, and
the mean was stable to four decimal places between the 70-clip and 199-clip partials. This
isn't drift on hard cases; it's the architecture.

Which is exactly what the config predicts. From ABot's own startup log:

```
CausalWanSelfAttention: max_attention_size=18480, local_attn_size=21, sink_size=0
```

**A 21-latent-frame attention window.** The mirror paths run 47–95 frames. By the time the
model is asked to reverse, the outbound half is outside the window — the information was
never *forgotten*, it was never in the computation. ABot also carries no depth model and no
3D cache; its checkpoint is a video VAE, a preview VAE, a text encoder, and a DiT.

## Scope and caveats

- **mirror_test only.** The full test set is 150 samples / ~350k frames ≈ 2.85 days ≈ $109
  on an A100. mirror_test clips are short (47–95 frames) *and* directly measure loop
  closure, so it's the best signal per dollar. 400 clips took 67 min.
- **Metrics run:** `lcm`, `visual`, `gsc`. **Skipped:** `action` (needs the ViPE conda env)
  and `dino` (DINOv3 is gated on HF — "requires approval").
- **Degraded runtime.** A100 (sm80), no FP8, SDPA instead of flash-attn. This affects
  *speed*, not output quality — the sampling path is unchanged.
- **Neutral prompt.** `"| unknown |"`, ABot's own benchmark default. MIND supplies no
  per-sample text, and this keeps it a test of action-conditioned memory rather than
  prompt-following.
- **Single seed, no repeats.** Error bars would need multiple seeds per clip.

## Reproducing

```
python mind_bridge.py inspect <mirror_test_dir>   # verify the action mapping
python mind_generate.py --anchors 20             # 400 clips, ~67 min
bash mind_eval.sh                                # lcm,visual,gsc
python summarize.py results/mind_abot.json
```

## The bug worth knowing about

**MIND encodes direction categorically, not by sign.** `0` = no input; `1` and `2` are the
two *opposing* directions on each axis:

```
ws  1→W  2→S      ad  1→A  2→D      ud  1→I  2→K      lr  1→J  2→L
```

Treating them as signed magnitudes — the obvious reading — collapses every opposing pair
onto the same key. `-a` and `-d` produce identical action sequences. That doesn't crash;
it produces plausible videos and a **completely meaningless memory score**, because a
benchmark that measures whether reversing returns you to the start would be sending the
same action both times.

It surfaced only because the inspection printed all ten paths side by side and two that
should have differed looked identical. `mind_bridge.py` documents the verified mapping.

## Environment traps

Five separate failures on the eval side alone, none about the model:

1. MIND pins `transformers==4.56.0`, ABot needs `5.4.0` → **separate venvs**.
2. `pip install` of MIND's ~50 deps died with `[Errno 5] Input/output error` on the
   network filesystem. Same install on the container disk: 50 seconds.
3. `process.py` iterates `['mem_test','action_space_test','mirror_test']` unconditionally
   and crashes on a missing dir → create empty siblings.
4. flash-attn 2.8.3 conflicts with xformers (`<=2.8.2`) and there's no 2.8.2 wheel for
   this stack → run on the SDPA fallback.
5. `pipeline_loader` resolves configs relative to cwd → must `chdir` to the repo root.
