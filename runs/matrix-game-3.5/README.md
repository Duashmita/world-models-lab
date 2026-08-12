# Matrix-Game 3.5 — Mughal city

Same prompt as `Trying/Marble1.1(t)`, for comparison — but note the tag difference.
Marble is **(t)**, text in. Matrix-Game 3.5 is **(p)**: it cannot take text, only an
image plus a camera path. Getting the same prompt through it needed an extra step.

## The prompt

> A photorealistic bustling mughal city with old Dravida architecture built as tall
> as skyscrappers at dusk in rural india by the lake

`prompt_expanded.txt` is the version actually fed to the model — MG-3.5's own sample
prompts are paragraph-length and specify material, light direction, and motion, and
short prompts underperform. Scene unchanged; added gopuram tiers, sandstone/granite,
ghats, boats, lantern glow, drifting smoke.

## Why there are two steps here (and only one in Marble)

Marble takes **text → 3D scene**. Matrix-Game 3.5 takes **image + camera path → video** —
it has no text-only entry point. So:

    prompt ──> Wan2.2-TI2V-5B ──> anchor frame ──> Matrix-Game 3.5 ──> mp4
                                  (1280x720)      + camera trajectory

Wan2.2 is not an extra model to install; MG-3.5 already requires it (T5 encoder, VAE,
DiT scaffold). Its `input_image` arg is optional, so it runs text-only for one frame.

MG-3.5 is a **continuation** model, not an origination model: it answers "where does
this camera go from here" and needs a "here". Any 1280x720 image works — a photo would
ground the geometry more reliably than a generated frame.

## Files

    anchor_raw_736.png    raw Wan2.2 output, seed 0. Wan rounds height to %32, so 720->736.
    anchor_720.png        center-cropped to 1280x720 so the sample intrinsics stay valid
                          (cx,cy = 639.05,359.05 assume 720, not 736)
    clips/city_case_N.mp4 the anchor flown through 6 camera paths
    cameras/cam_*.npz     purpose-built trajectories (orbit / crane / pan) — NOT yet rendered
    make_camera.py        trajectory generator, see below
    prompt_expanded.txt   what was actually sent

## About the clips

`city_case_0..5` reuse the camera paths bundled with MG-3.5's `samples/first_person/`,
which were authored for **other scenes** (mountains etc.). They're valid but all
forward-ish moves, differing mainly in travel distance, not shot type:

| clip | travel | note |
|---|---|---|
| case_1 | 16.8 u | fastest push-in |
| case_0 | 8.0 u  | |
| case_3 | 0.79 u | slowest, nearly static |

The `cameras/*.npz` are real alternative shots and would suit a lakefront city better.

## Camera trajectory format

Read off the bundled samples, which all agree:

- `extrinsics_c2w` — (N,4,4) float32 camera-to-world. **Frame 0 is identity.**
  Axes: +X right, +Y up, +Z forward.
- `intrinsics` — (N,4) `[fx,fy,cx,cy]`, **constant across frames**. `fx` sets FOV:
  647 ≈ 89° wide, 794 ≈ 78°, 1078 ≈ 61° tele.
- Scale: samples travel 0.8–17 units over 961 frames; per-frame step 0.002–0.03.
  ~1 unit ≈ a room. Large steps tear the generation.

Generate paths with:

    python make_camera.py out.npz orbit --radius 5 --deg 75
    python make_camera.py out.npz crane --dist 4 --rise 3 --deg -18
    python make_camera.py out.npz pan   --deg 70 --fx 1078
    # also: dolly, truck, pedestal, tilt

**Unverified:** every bundled path keeps `y=0`, so whether +Y renders as up or down
is untested. If a crane/tilt comes out inverted, flip the sign on `--deg`.

## Reproducing

On the pod (`m24fyk9980huz7`, see ../../STATUS.md):

    bash /workspace/gen_anchor.sh                 # prompt -> anchor, ~40 s
    bash /workspace/batch_city.sh                 # 6 clips, ~3 min each

Both free the GPU from the ABot UI first — ABot idles at ~22 GB and MG-3.5 peaks at
~67 GB, so they cannot share the A100.
