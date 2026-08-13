#!/usr/bin/env python3
"""Summarise MIND results for ABot-World."""
import json, sys, statistics as st
from collections import defaultdict

d = json.load(open(sys.argv[1]))
rows = d.get("data", [])
print(f"samples in file: {len(rows)}")
errs = [r for r in rows if r.get("error")]
if errs:
    print(f"  with errors: {len(errs)}  e.g. {errs[0].get('error')!r}")

# gsc lives per-video under video_results — this is the loop-closure metric
gsc = defaultdict(list)
by_persp = defaultdict(lambda: defaultdict(list))
nvid = 0
for r in rows:
    persp = r.get("perspective", "?")
    for vr in r.get("video_results", []) or []:
        g = vr.get("gsc") or {}
        if not g:
            continue
        nvid += 1
        for k in ("avg_mse", "avg_lpips", "avg_ssim", "avg_psnr"):
            v = g.get(k)
            if isinstance(v, (int, float)):
                gsc[k].append(v)
                by_persp[persp][k].append(v)

print(f"\n=== gsc (loop closure: outbound vs reversed return), n={nvid} videos ===")
for k in ("avg_mse", "avg_lpips", "avg_ssim", "avg_psnr"):
    v = gsc[k]
    if v:
        print(f"  {k:10s} mean={st.mean(v):8.4f}  median={st.median(v):8.4f}  "
              f"min={min(v):8.4f}  max={max(v):8.4f}")

print("\n=== gsc by perspective ===")
for persp in sorted(by_persp):
    row = by_persp[persp]
    parts = [f"{k.replace('avg_',''):5s}={st.mean(v):7.4f}" for k, v in row.items() if v]
    print(f"  {persp:10s} n={len(row.get('avg_mse', [])):4d}  " + "  ".join(parts))

# lcm / visual, if present at sample level
for key, label in (("lcm", "lcm (vs ground truth)"), ("visual_quality", "visual quality")):
    acc = defaultdict(list)
    for r in rows:
        m = r.get(key) or {}
        for k, v in m.items():
            if isinstance(v, (int, float)) and k.startswith("avg_"):
                acc[k].append(v)
    if acc:
        print(f"\n=== {label} ===")
        for k, v in acc.items():
            print(f"  {k:14s} mean={st.mean(v):8.4f}  n={len(v)}")
