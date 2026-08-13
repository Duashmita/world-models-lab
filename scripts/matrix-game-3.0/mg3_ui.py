#!/usr/bin/env python3
"""Minimal web UI for Matrix-Game 3.0.

MG-3.0 ships no UI — only a terminal REPL that reads two input() calls per step and
writes an mp4 per action with no on-screen feedback, so it feels dead while working.

This does NOT refactor the pipeline. It runs generate.py unmodified with stdin attached
to a FIFO, writes action pairs into that FIFO when you press a button, and plays back
each clip as it lands. The FIFO writer is held open for the process lifetime — MG-3.0's
get_current_action() wraps input() in a bare `except: pass`, so an EOF would spin forever.
"""
import os, subprocess, threading, time, glob, signal, fcntl
import gradio as gr

REPO = "/workspace/repos/Matrix-Game/Matrix-Game-3"
VENV = "/workspace/venvs/mg3/bin/python"
CKPT = "/workspace/ckpt/mg3"
OUT = "/workspace/output"
FIFO = "/workspace/mg_ui_fifo"

CAMERA = [("look up", "i"), ("look down", "k"), ("look left", "j"),
          ("look right", "l"), ("hold", "u")]
MOVE = [("forward", "w"), ("back", "s"), ("strafe left", "a"),
        ("strafe right", "d"), ("stand", "q")]

state = {"proc": None, "fifo": None, "log": [], "started": 0.0}


def _log(msg):
    state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    state["log"] = state["log"][-200:]


def _reader(pipe):
    for line in iter(pipe.readline, b""):
        s = line.decode("utf-8", "replace").rstrip()
        el = state.get("enginelog")
        if el:
            el.write(s + "\n"); el.flush()
        if s and "Please input" not in s:
            _log(s[:160])
    pipe.close()


def start_engine(image_path, prompt):
    if state["proc"] and state["proc"].poll() is None:
        return "already running", _latest(), _logs()
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "ui_*.mp4")):
        os.remove(f)
    if os.path.exists(FIFO):
        os.remove(FIFO)
    os.mkfifo(FIFO)

    img = image_path or os.path.join(REPO, "demo_images/001/image.png")
    cmd = [VENV, "-u", "generate.py", "--size", "704*1280", "--ckpt_dir", CKPT,
           "--output_dir", OUT, "--fa_version", "2", "--num_iterations", "200",
           "--num_inference_steps", "3", "--image", img,
           "--prompt", prompt or "A colorful, animated cityscape.",
           "--save_name", "ui", "--seed", "42", "--vae_type", "mg_lightvae",
           "--interactive"]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    # Open the read end non-blocking so we don't deadlock waiting for a writer, then
    # CLEAR O_NONBLOCK before handing it to the child. A non-blocking stdin makes
    # input() raise instead of wait, and MG-3.0's bare `except: pass` turns that into
    # an infinite spin that never loads the model.
    fin = os.open(FIFO, os.O_RDONLY | os.O_NONBLOCK)
    flags = fcntl.fcntl(fin, fcntl.F_GETFL)
    fcntl.fcntl(fin, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    enginelog = open("/workspace/logs/engine.log", "w")
    state["proc"] = subprocess.Popen(cmd, cwd=REPO, stdin=fin,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=env, preexec_fn=os.setsid)
    os.close(fin)
    # hold the write end open forever so input() never sees EOF
    state["fifo"] = open(FIFO, "w")
    state["enginelog"] = enginelog
    state["started"] = time.time()
    threading.Thread(target=_reader, args=(state["proc"].stdout,), daemon=True).start()
    _log("engine starting — first action waits ~60s for model load")
    return "starting…", _latest(), _logs()


def step(cam_label, move_label):
    if not (state["proc"] and state["proc"].poll() is None):
        return "engine not running — press Start", _latest(), _logs()
    cam = dict(CAMERA)[cam_label]
    mv = dict(MOVE)[move_label]
    try:
        state["fifo"].write(f"{cam}\n{mv}\n")
        state["fifo"].flush()
    except Exception as e:
        return f"write failed: {e}", _latest(), _logs()
    _log(f"action  camera={cam.upper()}  move={mv.upper()}")
    return f"sent {cam.upper()}+{mv.upper()} — clip appears in ~10-20s", _latest(), _logs()


def _latest():
    vids = sorted(glob.glob(os.path.join(OUT, "ui_*.mp4")), key=os.path.getmtime)
    return vids[-1] if vids else None


def _logs():
    return "\n".join(state["log"][-18:])


def refresh():
    n = len(glob.glob(os.path.join(OUT, "ui_*.mp4")))
    alive = state["proc"] and state["proc"].poll() is None
    up = int(time.time() - state["started"]) if state["started"] else 0
    return (f"{'running' if alive else 'stopped'} · {n} clips · {up}s",
            _latest(), _logs())


def stop_engine():
    if state["proc"] and state["proc"].poll() is None:
        os.killpg(os.getpgid(state["proc"].pid), signal.SIGKILL)
        _log("engine stopped")
    if state["fifo"]:
        try: state["fifo"].close()
        except Exception: pass
    return "stopped", _latest(), _logs()


with gr.Blocks(title="Matrix-Game 3.0") as demo:
    gr.Markdown("## Matrix-Game 3.0 — interactive\n"
                "Pick a camera move and a movement, then **Step**. "
                "Each step generates ~57 frames (~3.3 s). First step waits for model load.")
    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Video(label="latest clip", autoplay=True, height=420)
        with gr.Column(scale=2):
            status = gr.Textbox(label="status", value="not started", interactive=False)
            cam = gr.Radio([c[0] for c in CAMERA], value="hold", label="camera")
            mv = gr.Radio([m[0] for m in MOVE], value="forward", label="movement")
            step_btn = gr.Button("Step ▶", variant="primary")
            with gr.Row():
                start_btn = gr.Button("Start engine")
                stop_btn = gr.Button("Stop", variant="stop")
            refresh_btn = gr.Button("Refresh")
    with gr.Accordion("scene (applies on Start)", open=False):
        img_in = gr.Textbox(label="image path", value="", placeholder="blank = bundled demo")
        prompt_in = gr.Textbox(label="prompt", value="A colorful, animated cityscape.", lines=3)
    logbox = gr.Textbox(label="engine log", lines=10, interactive=False)

    outs = [status, video, logbox]
    start_btn.click(start_engine, [img_in, prompt_in], outs)
    step_btn.click(step, [cam, mv], outs)
    stop_btn.click(stop_engine, None, outs)
    refresh_btn.click(refresh, None, outs)
    gr.Timer(5).tick(refresh, None, outs)

# allowed_paths: gradio refuses to serve files outside its own temp dirs, so the
# generated clips in /workspace/output are invisible to the Video component without this.
demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[OUT])
