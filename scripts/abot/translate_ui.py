#!/usr/bin/env python3
"""Translate ABot-World's Gradio UI from Chinese to English.

Only touches user-visible strings — button labels, markdown, status text, gallery
labels, key-HUD captions. Code comments are left alone (they don't render, and
rewriting them would churn the diff for no benefit).

Backs up every file it edits to <name>.zh.bak.
"""
import os, shutil, sys

WEB = "/workspace/repos/ABot-World/web_client"

# (filename, [(chinese, english), ...])
EDITS = {
    "app.py": [
        ('title="ABot-World - 实时可交互世界模型"',
         'title="ABot-World - Real-Time Interactive World Model"'),
        ('正在链接你的世界', 'Connecting to your world'),
        ('label="描绘你的世界"', 'label="Describe your world"'),
        ('gr.Button("唤醒你的世界"', 'gr.Button("Awaken your world"'),
        ('gr.Button("封存你的世界"', 'gr.Button("Seal your world"'),
        ('"**探索平行宇宙**"', '"**Explore parallel worlds**"'),
        ('"**[仅前端调试]** 未加载模型，界面可正常操作。"',
         '"**[Frontend debug only]** Model not loaded; the interface still works."'),
        ('"**模型加载中…** 加载完成后「唤醒你的世界」将可用。"',
         '"**Loading model…** \'Awaken your world\' will be enabled when it finishes."'),
    ],
    "ui_helpers.py": [
        ('f"**当前 Prompt**: {line}"', 'f"**Current prompt**: {line}"'),
        ("f\"已生成: {data['block_count']} blocks | {data['frame_count']} 帧 | {data['elapsed']:.1f}s\\n\"",
         "f\"Generated: {data['block_count']} blocks | {data['frame_count']} frames | {data['elapsed']:.1f}s\\n\""),
        ("f\"推理速度: {data['bps']:.2f} blocks/s | {data['fps']:.1f} fps\"",
         "f\"Inference speed: {data['bps']:.2f} blocks/s | {data['fps']:.1f} fps\""),
        ('已播放 {played}/{total} 帧', 'Played {played}/{total} frames'),
        ('已生成 {generated}/{total} 帧', 'Generated {generated}/{total} frames'),
    ],
    "inference.py": [
        ('value="唤醒你的世界"', 'value="Awaken your world"'),
        ('value="停止中..."', 'value="Stopping..."'),
        ('value="重塑你的世界"', 'value="Reshape your world"'),
        ('f"**错误**：{state.worker_error}"', 'f"**Error**: {state.worker_error}"'),
        ('f"**错误**：{e}"', 'f"**Error**: {e}"'),
        ('gr.Info("✅ 视频生成完成"', 'gr.Info("✅ Video generation complete"'),
        ('"**[仅前端调试]** 未加载模型，界面可正常操作。"',
         '"**[Frontend debug only]** Model not loaded; the interface still works."'),
        ('"**就绪** — 请输入 Prompt 并点击「唤醒你的世界」。"',
         '"**Ready** — enter a prompt and click \'Awaken your world\'."'),
        ('"**模型加载中…** 加载完成后「唤醒你的世界」将可用。"',
         '"**Loading model…** \'Awaken your world\' will be enabled when it finishes."'),
        ('"**[仅前端调试]** 未加载模型，点击无实际效果。"',
         '"**[Frontend debug only]** Model not loaded; this button does nothing."'),
        ('"**模型加载中…** 请稍候再试。"', '"**Loading model…** please wait and retry."'),
        ('"**准备中…**"', '"**Preparing...**"'),
    ],
    "scene_presets.yaml": [
        ('label: 城郊小院', 'label: Suburban yard'),
        ('label: 004_第三人称沙漠河谷行走', 'label: 004 Third-person desert valley walk'),
        ('label: 007_第一人称高山草甸小径', 'label: 007 First-person alpine meadow path'),
        ('label: 010_第一人称林间溪流踏石', 'label: 010 First-person forest stream stepping-stones'),
    ],
    "key_hud_wasd.html": [('移动', 'MOVE')],
    "key_hud_ijkl.html": [('视角', 'LOOK')],
}

total = 0
for fn, pairs in EDITS.items():
    path = os.path.join(WEB, fn)
    if not os.path.exists(path):
        print(f"  !! missing {fn}"); continue
    src = open(path, encoding="utf-8").read()
    bak = path + ".zh.bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    hits = 0
    for zh, en in pairs:
        if zh in src:
            src = src.replace(zh, en); hits += 1
        else:
            print(f"  ?? not found in {fn}: {zh[:40]}")
    open(path, "w", encoding="utf-8").write(src)
    print(f"  {fn}: {hits}/{len(pairs)} replaced")
    total += hits

print(f"\ntotal replacements: {total}")

# syntax check the python files we touched
import ast
for fn in ("app.py", "ui_helpers.py", "inference.py"):
    p = os.path.join(WEB, fn)
    try:
        ast.parse(open(p, encoding="utf-8").read())
        print(f"  syntax OK  {fn}")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in {fn}: {e}"); sys.exit(1)

import yaml
try:
    yaml.safe_load(open(os.path.join(WEB, "scene_presets.yaml"), encoding="utf-8"))
    print("  yaml OK    scene_presets.yaml")
except Exception as e:
    print("  YAML ERROR:", e); sys.exit(1)
