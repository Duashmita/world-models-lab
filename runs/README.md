# Runs

Same prompt through every model. Nothing here is cherry-picked — the static and failed
clips are kept, because they're what the log is about.

## `marble/`
World Labs Marble 1.1, text straight in. A **Renderer**: it returns persistent, exportable
3D geometry rather than frames. The only thing here that keeps a scene.

## `wan2.2/`
`wan_city_t2v.mp4` — Wan2.2-TI2V-5B, pure text-to-video. **No anchor image, no camera
path.** 121 frames @ 24 fps, 5.0 s. The control condition for "does the scene animate on
its own": whatever moves here, the model chose to move.

## `matrix-game-3.5/`
- `anchor_720.png` — generated from the prompt with Wan2.2, then center-cropped from
  1280×736 to 1280×720 so the sample camera intrinsics (cx,cy = 639.05, 359.05) stay valid.
- `city_case_0..5.mp4` — **the mistake.** Rendered with the default `num_blocks: 1`, so
  85 frames from a 961-frame camera path. They barely move. Kept deliberately.
- `palace_tour_6b.mp4` — the fix. 6 blocks, 505 frames, 31.6 s, on a purpose-built
  waterfront trajectory. Measured first-vs-last frame difference **61.4/255** against
  **16.5** for the 1-block clips: ~3.7× more movement.
- `cam_*.npz` — camera trajectories built by `scripts/matrix-game-3.5/make_camera.py`,
  validated against the bundled samples but never rendered.

## `matrix-game-3.0/`
Four clips from the interactive REPL, one per action step. 1280×704, 57 frames @ 17 fps
≈ 3.35 s each. Actions were `U` (no camera move) + `W` (forward).

## `abot-world/`
- `abot_city_walk.mp4` — offline batch rollout, 477 frames / 39.8 s, driven by a scripted
  40-block action sequence.
- `session_3min.mp4` and `session_10min.mp4` — live Gradio sessions, WASD-driven.
  **The 10-minute one is the interesting artifact**: 7,197 frames at 12 fps, long enough
  to watch memory degrade in real time. Recompressed (CRF 30) to fit GitHub; the original
  was 230 MB.
- `stream_*.png` — the anchor frame each session started from.

## Reading the outputs fairly

Everything ran on Ampere with no FP8 and, for ABot, a stubbed quantisation path. Quality
comparisons hold — same prompt, same anchor. **Speed comparisons do not.**

Clip length varies by architecture and is not a quality signal: Wan2.2 is capped at 5 s by
its frame budget, Matrix-Game 3.0 emits ~3.3 s per action, ABot rolls indefinitely.
