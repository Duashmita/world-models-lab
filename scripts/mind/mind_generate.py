#!/usr/bin/env python3
"""Generate ABot-World predictions for MIND's mirror_test, in MIND's expected layout.

ABot's own scripts/inference.py loads the model per invocation (~90 s). For 200 clips
that is 5+ hours of pure reload. get_pipeline() is cached and set_first_frame_latent()
takes the anchor per call, so we load once and loop.

Output layout MIND expects:
    {model}/{persp}_data/mirror_test/{anchor}/path-1..10.mp4

Each mirror path is out-and-back by construction (see mind_bridge.py), which is what
MIND's `gsc` scores: it flips the second half and compares it to the first.
"""
import os, sys, json, glob, time, argparse, traceback

ABOT = "/workspace/repos/ABot-World"
sys.path.insert(0, ABOT)
sys.path.insert(0, "/workspace")
# pipeline_loader resolves configs/ and checkpoints/ relative to cwd, not to the module,
# so it must run from the repo root or OmegaConf looks for /root/configs/...
os.chdir(ABOT)

import torch
from mind_bridge import to_block_actions, KEY_ORDER

from web_client.config import KEY_ORDER as ABOT_KEYS, STREAM_HEIGHT, STREAM_WIDTH
from web_client.pipeline_loader import get_pipeline, decode_block_to_frames
import imageio

assert list(ABOT_KEYS) == list(KEY_ORDER), f"key order mismatch: {ABOT_KEYS} vs {KEY_ORDER}"

VIDEO_FPS = 16


def write_video(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = imageio.get_writer(path, fps=VIDEO_FPS, codec="libx264", quality=8,
                           macro_block_size=1, ffmpeg_log_level="error")
    for f in frames:
        w.append_data(f)
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", default="/workspace/data/MIND-Data")  # absolute: we chdir'd
    ap.add_argument("--out-root", default="/workspace/preds/ABot-World")
    ap.add_argument("--anchors", type=int, default=10, help="anchors per perspective")
    ap.add_argument("--perspectives", default="1st_data,3rd_data")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (0=all)")
    ap.add_argument("--prompt", default="| unknown |", help="ABot benchmark default")
    a = ap.parse_args()

    print("loading ABot pipeline (once)...", flush=True)
    t0 = time.time()
    pipeline, _, device = get_pipeline(vae_type=None, use_fp8_gemm=False, quant_type="none")
    num_fpb = int(pipeline.num_frame_per_block)
    frames_per_block = num_fpb * 4
    vae = pipeline.encoder if pipeline.encoder is not None else pipeline.vae
    up = getattr(vae, "upsampling_factor", 8)
    latent_shape = (1, num_fpb, vae.z_dim, STREAM_HEIGHT // up, STREAM_WIDTH // up)
    print(f"  loaded in {time.time()-t0:.0f}s | {num_fpb} latents/block "
          f"= {frames_per_block} frames/block", flush=True)

    # set_prompts() must precede set_first_frame_latent() or the pipeline raises
    # "call set_prompts first". It also resets the cross-attention cache, so call it once.
    # MIND supplies no per-sample text; "| unknown |" is ABot's own benchmark default and
    # keeps this a test of action-conditioned memory rather than prompt-following.
    pipeline.set_prompts([a.prompt], device=device)
    print(f"  prompt set: {a.prompt!r}", flush=True)

    # build the job list
    jobs = []
    for persp in a.perspectives.split(","):
        mdir = os.path.join(a.gt_root, persp, "test", "mirror_test")
        if not os.path.isdir(mdir):
            print("  missing", mdir); continue
        paths = sorted(glob.glob(os.path.join(mdir, "*.json")))
        anchors = sorted(glob.glob(os.path.join(mdir, "*.png")))[:a.anchors]
        for anc in anchors:
            name = os.path.splitext(os.path.basename(anc))[0]
            for i, pj in enumerate(paths, start=1):
                jobs.append((persp, name, anc, pj, i))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"  {len(jobs)} clips to generate", flush=True)

    manifest, done, t_start = [], 0, time.time()
    for persp, name, anchor, path_json, idx in jobs:
        out = os.path.join(a.out_root, persp, "mirror_test", name, f"path-{idx}.mp4")
        if os.path.exists(out):
            done += 1; continue
        try:
            blocks, total, mark = to_block_actions(path_json, frames_per_block)
            frames = []
            for bi, act in enumerate(blocks):
                noise = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)
                if bi == 0:
                    pipeline.set_first_frame_latent(anchor, height=STREAM_HEIGHT,
                                                    width=STREAM_WIDTH, device=device)
                pipeline.set_act({k: bool(v) for k, v in zip(KEY_ORDER, act)},
                                 height=STREAM_HEIGHT, width=STREAM_WIDTH,
                                 num_frames=num_fpb, device=device)
                pred = pipeline.generate_next_block(noise)
                if pred is None:
                    continue
                frames.extend(decode_block_to_frames(pipeline, pred))
            if not frames:
                raise RuntimeError("no frames produced")
            write_video(frames, out)
            manifest.append({"perspective": persp, "anchor": name,
                             "path_json": os.path.basename(path_json), "path_index": idx,
                             "gt_total_time": total, "gt_mark_time": mark,
                             "blocks": len(blocks), "frames": len(frames), "out": out})
            done += 1
            el = time.time() - t_start
            rate = done / el * 60 if el else 0
            eta = (len(jobs) - done) / rate if rate else 0
            print(f"  [{done}/{len(jobs)}] {persp}/{name}/path-{idx} "
                  f"{len(frames)}f  {rate:.1f} clips/min  eta {eta:.0f}min", flush=True)
        except Exception as e:
            print(f"  FAILED {persp}/{name}/path-{idx}: {e}", flush=True)
            traceback.print_exc()

    os.makedirs(a.out_root, exist_ok=True)
    with open(os.path.join(a.out_root, "manifest.json"), "w") as f:
        json.dump({"clips": manifest, "frames_per_block": frames_per_block}, f, indent=2)
    print(f"DONE {done}/{len(jobs)} in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
