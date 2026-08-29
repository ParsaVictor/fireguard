# Builds FireGuard.ipynb from the tested code in fireguard_core.py
import json
import re

SRC = open("fireguard_core.py", encoding="utf-8").read()

SECTION = re.compile(r"# ={10,}\s*\n#\s*(.+?)\s*\n# ={10,}\s*\n")
chunks = SECTION.split(SRC)
# chunks: [imports, "CONFIG — ...", code, "MODEL REGISTRY — ...", code, ...]
titles = [t for t in chunks[1::2]]
bodies = chunks[2::2]
print("sections:", titles)
imports_c = chunks[0].strip("\n")
by_title = {t.split(" ")[0].split("—")[0].strip(): b.strip("\n") for t, b in zip(titles, bodies)}
config_c = by_title["CONFIG"]
registry_c = by_title["MODEL"]
detect_c = by_title["DETECTION"]
hazard_c = by_title["HAZARD"]
overlay_c = by_title["OVERLAY"]
pipeline_c = by_title["PIPELINE"].split('if __name__ == "__main__":')[0].strip("\n")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").split("\n")}


TITLE_MD = """
# 🔥 FireGuard — Real-Time Fire & Smoke Detection

**Broadcast-grade fire & smoke intelligence on video — three generations of YOLO, one pipeline.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/fireguard/blob/main/FireGuard.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![ultralytics](https://img.shields.io/badge/YOLO-ultralytics-8A2BE2.svg)]()

### ✨ What you get
| Capability | How |
|---|---|
| 🎯 **Accurate detection** | Swappable pretrained models: **YOLOv8n / YOLO11s / YOLO26s** (no training needed) |
| 🟥 **Region marking** | Fire area is filled, bracketed, labelled and wrapped in a dashed **safety zone** |
| 🚦 **Hazard state machine** | `SAFE → CAUTION → WARNING → DANGER → CRITICAL` with temporal confirmation (no flicker, no false alarms) |
| 📢 **Smart alerts** | Anti-spam cooldown, `events.csv` timeline, evidence snapshots on every escalation |
| ⚡ **Fast & light** | Per-class confidence, frame-skip, FP16 on GPU, CPU-friendly by default |
| 🧩 **Profiles** | `standard` / `fire-only` (government mode) / `early-smoke` (earliest warning) |

### 🇮🇷 فارسی
سیستم هوشمند تشخیص آتش و دود روی ویدیو — سه نسل مدل YOLO به‌صورت قابل تعویض، مشخص‌سازی دقیق محدوده آتش، ماشین وضعیت خطر با تأیید زمانی، هشدارهای هوشمند و خروجی ویدیویی حرفه‌ای. کافیست ویدیو را معرفی کنید و سلول‌ها را اجرا کنید.

---
"""

SETUP_MD = "## 1 · Setup\nInstalls everything (safe to re-run — skips what's already present). In **Colab**, enable GPU: `Runtime → Change runtime type → T4 GPU`."
SETUP_CODE = """
%pip install -q ultralytics huggingface_hub matplotlib pandas
"""

IMPORTS_MD = "## 2 · Imports & device"
CONFIG_MD = """## 3 · Configuration — *the one cell you edit*

| Knob | Meaning |
|---|---|
| `model` | `yolov8n` (fastest, 3.0M params) · `yolo11s` (balanced) · `yolo26s` (newest, most stable) · or a path to **any** custom `.pt` |
| `profile` | `standard` · `fire-only` (ignore smoke — e.g. government/open-terrain) · `early-smoke` (lowest smoke threshold) |
| `conf_fire` / `conf_smoke` | `None` → taken from the profile; set a number to override |
| `frame_skip` | run inference every Nth frame (2 ≈ 2× faster, boxes are reused between) |
| `confirm_window` / `confirm_hits` / `clear_frames` | temporal hysteresis — how stubborn the alarms are |
"""
REGISTRY_MD = "## 4 · Model zoo — three YOLO generations\nAll weights are **verified, public Hugging Face checkpoints** (downloaded once, cached in `models/`)."
DETECT_MD = "## 5 · Detection layer\nNormalises every dataset's label spelling (`Fire`, `fire`, `FLAME`…) and applies per-class confidence + minimum-box-size filtering."
HAZARD_MD = "## 6 · Hazard engine — state machine with hysteresis\nA fire alarm must be **confirmed over several frames** (no one-frame false positives) and **survives brief dropout** (no flicker)."
OVERLAY_MD = "## 7 · Overlay renderer\nCorner brackets · label chips · translucent fire-region fill · dashed safety zone · HUD · pulsing border."
PIPELINE_MD = "## 8 · Video pipeline\nRead → detect → confirm → annotate → write. Produces the annotated MP4 (H.264), `events.csv` and escalation snapshots."
VIZ_MD = "## 9 · Visualisation helpers"
RUN_MD = "## 10 · Run it ▶️"
RUN_CODE = """
import os

try:                                     # canonical Colab detection
    from google.colab import drive       # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive")
    VIDEO_PATH = "/content/drive/MyDrive/fire_5.mp4"        # ← EDIT: your video in Drive
else:
    VIDEO_PATH = "fire_detection_result.mp4"                # ← EDIT: local path

# Demo convenience: analyse just the fire-heavy section of the bundled demo video.
# For YOUR video set both to 0 / None (whole video).
cfg = dict(CONFIG)
if VIDEO_PATH.endswith("fire_detection_result.mp4"):
    cfg["start_frame"], cfg["max_frames"] = 3400, 200
else:
    cfg["start_frame"], cfg["max_frames"] = 0, None

assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
model = load_model(CONFIG["model"])

summary, samples = process_video(model, cfg, VIDEO_PATH, out_name="fireguard_result.mp4")
show_samples(samples)
plot_timeline(summary)
show_events()
"""
VIZ_CODE = '''
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

STATE_ORDER = ["SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"]


def _hex(bgr):
    return "#%02x%02x%02x" % tuple(reversed(bgr))


def show_samples(samples, cols=3):
    """Annotated frames in a grid (BGR -> RGB)."""
    if not samples:
        print("no samples collected")
        return
    rows = -(-len(samples) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, img in zip(axes, samples):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


def plot_timeline(summary):
    """Hazard state over time — the whole incident at a glance."""
    df = pd.DataFrame(summary["timeline"])
    if df.empty:
        return
    df["state_idx"] = df["state"].map({s: i for i, s in enumerate(STATE_ORDER)})
    fig, ax = plt.subplots(figsize=(11, 3))
    for s in STATE_ORDER:
        y = df["state_idx"].where(df["state"] == s, np.nan)   # NaN -> gap, no broadcast issue
        if y.notna().any():
            ax.fill_between(df["time_s"], 0, y, step="post",
                            alpha=0.35, color=_hex(HAZARD_STATES[s]["color"]), label=s)
    ax.step(df["time_s"], df["state_idx"], where="post", color="white", lw=1.2)
    ax.set_yticks(range(len(STATE_ORDER)), STATE_ORDER)
    ax.set_xlabel("time (s)")
    ax.set_title("Hazard state over time")
    ax.legend(loc="upper right", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.show()


def show_events():
    p = os.path.join(CONFIG["output_dir"], "events.csv")
    if os.path.exists(p):
        print("📄 events.csv")
        display(pd.read_csv(p))
    else:
        print("📄 no state changes were recorded (video stayed SAFE)")
'''
BENCH_MD = "## 11 · Benchmark — all three models on the same section\nSpeed (CPU here — expect **10–30× faster on GPU**) and behavioural stability (fewer state changes = more temporally stable detections)."
BENCH_CODE = """
rows = []
for key in MODEL_REGISTRY:
    m = load_model(key)
    c = dict(cfg)
    c["display_samples"], c["snapshot"] = 0, False
    c["max_frames"] = min(c["max_frames"] or 120, 120)
    s, _ = process_video(m, c, VIDEO_PATH, out_name=f"bench_{key}.mp4", quiet=True)
    rows.append({
        "model": key,
        "params": MODEL_REGISTRY[key]["params"],
        "ms/frame": round(s["ms_per_inference"], 1),
        "FPS (inference)": round(1000 / s["ms_per_inference"], 1),
        "state changes": s["events"],
        "fire seen": s["fire_seen"],
        "smoke seen": s["smoke_seen"],
    })

bench = pd.DataFrame(rows)
os.makedirs(CONFIG["output_dir"], exist_ok=True)
bench.to_csv(os.path.join(CONFIG["output_dir"], "benchmark.csv"), index=False)
bench
"""
EXTRAS_MD = """
## 12 · Going further

- **Custom weights** — trained your own model? Just set `CONFIG["model"] = "path/to/best.pt"`; any Ultralytics detect model works, label spelling is auto-normalised.
- **Real-time camera** — replace the video path with `0` in OpenCV and stream through the same pipeline.
- **Deployment** — export to ONNX / TensorRT: `model.export(format="onnx")` for edge devices (Jetson, RPi5 + Hailo).
- **Alerting** — hook `events.csv` / snapshots into a Telegram bot, webhook or SIEM.
- **Roadmap** — RTSP multi-camera manager · object tracking IDs (how many distinct fires) · burned-area estimation · Gradio web demo.

---

*FireGuard is an assistive tool — always comply with local fire-safety regulations; do not use it as the sole life-safety system.*
"""

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "cells": [
        md(TITLE_MD),
        md(SETUP_MD), code(SETUP_CODE),
        md(IMPORTS_MD), code(imports_c),
        md(CONFIG_MD), code(config_c),
        md(REGISTRY_MD), code(registry_c),
        md(DETECT_MD), code(detect_c),
        md(HAZARD_MD), code(hazard_c),
        md(OVERLAY_MD), code(overlay_c),
        md(PIPELINE_MD), code(pipeline_c),
        md(VIZ_MD), code(VIZ_CODE),
        md(RUN_MD), code(RUN_CODE),
        md(BENCH_MD), code(BENCH_CODE),
        md(EXTRAS_MD),
    ],
}

# nbformat wants each source line to end with \n (except the last)
for cell in nb["cells"]:
    src = cell["source"]
    cell["source"] = [l if l.endswith("\n") else l + "\n" for l in src]

with open("FireGuard.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("FireGuard.ipynb written:", sum(1 for c in nb["cells"] if c["cell_type"] == "code"), "code cells")
