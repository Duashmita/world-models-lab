#!/usr/bin/env python3
"""MIND -> ABot-World action bridge.

MIND action.json, per frame:
    {"time": t, "ws": .., "ad": .., "ud": .., "lr": ..,
     "actor_pos": {...}, "actor_rpy": {...}}

  ws  forward/back      ad  strafe left/right
  ud  look up/down      lr  look left/right

ABot takes one 8-vector per *block* (a block is ~12 generated frames):
    [W, A, S, D, I, J, K, L]
    W/A/S/D movement, I/K look up/down, J/L look left/right

So we downsample MIND's per-frame actions to per-block by majority vote inside each
block window, then binarise by sign.

Usage:
  python mind_bridge.py inspect <mirror_test_dir>
  python mind_bridge.py convert <action.json> [frames_per_block]
"""
import json, sys, os, glob
from collections import Counter

KEY_ORDER = ["W", "A", "S", "D", "I", "J", "K", "L"]
FRAMES_PER_BLOCK = 12


# MIND encodes direction CATEGORICALLY, not by sign: 0 = no input, 1 and 2 are the two
# opposing directions on that axis. Treating them as magnitudes collapses every opposing
# pair onto the same key (-a and -d both became "D"), which would silently invalidate the
# whole benchmark.
#
# Verified against the mirror_test paths, whose filenames encode ground truth and whose
# temporal order is always out-then-back:
#   -w:  ws1 x24 -> ws2 x24        -a: ad1 x24 -> ad2 x24
#   -s:  ws2 x24 -> ws1 x24        -d: ad2 x24 -> ad1 x24
#   -u:  ud1 -> ud2                -down: ud2 -> ud1
#   -wl: ws1 -> lr1 -> lr2 -> ws2  -wr:   ws1 -> lr2 -> lr1 -> ws2
AXIS_MAP = {
    "ws": {1: "W", 2: "S"},   # forward / back
    "ad": {1: "A", 2: "D"},   # strafe left / right
    "ud": {1: "I", 2: "K"},   # look up / down
    "lr": {1: "J", 2: "L"},   # look left / right
}


def frame_to_keys(e):
    """One MIND frame -> set of ABot keys."""
    keys = set()
    for axis, table in AXIS_MAP.items():
        k = table.get(e.get(axis, 0))
        if k:
            keys.add(k)
    return keys


def to_block_actions(action_json_path, frames_per_block=FRAMES_PER_BLOCK):
    """MIND action.json -> (list of 8-vectors, total_time, mark_time).

    Majority vote per key within each block window: a key is pressed for the block if
    it's active in >= half the frames of that window. This preserves sustained motion
    and drops single-frame jitter.
    """
    d = json.load(open(action_json_path))
    data = d["data"]
    total, mark = d["total_time"], d["mark_time"]
    n_blocks = max(1, (len(data) + frames_per_block - 1) // frames_per_block)
    blocks = []
    for b in range(n_blocks):
        window = data[b * frames_per_block:(b + 1) * frames_per_block]
        if not window:
            break
        counts = Counter()
        for e in window:
            for k in frame_to_keys(e):
                counts[k] += 1
        half = len(window) / 2.0
        blocks.append([1 if counts[k] >= half else 0 for k in KEY_ORDER])
    return blocks, total, mark


def describe(vec):
    return "".join(k if v else "." for k, v in zip(KEY_ORDER, vec))


def cmd_inspect(d):
    js = sorted(glob.glob(os.path.join(d, "*.json")))
    pngs = sorted(glob.glob(os.path.join(d, "*.png")))
    print("anchors: %d   action paths: %d" % (len(pngs), len(js)))
    print()
    print("%-12s %5s %5s %7s  %s" % ("path", "n", "mark", "blocks", "block actions (WASDIJKL)"))
    for f in js:
        blocks, total, mark = to_block_actions(f)
        name = os.path.basename(f)
        strs = " ".join(describe(b) for b in blocks[:8])
        print("%-12s %5d %5d %7d  %s" % (name, total, mark, len(blocks), strs))
    print()
    print("first 3 anchors:", [os.path.basename(p) for p in pngs[:3]])


def cmd_convert(f, fpb=FRAMES_PER_BLOCK):
    blocks, total, mark = to_block_actions(f, int(fpb))
    print(json.dumps({
        "source": f, "total_time": total, "mark_time": mark,
        "frames_per_block": int(fpb), "n_blocks": len(blocks),
        "action_list": blocks,
    }))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "inspect":
        cmd_inspect(sys.argv[2])
    elif sys.argv[1] == "convert":
        cmd_convert(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else FRAMES_PER_BLOCK)
    else:
        print(__doc__); sys.exit(1)
