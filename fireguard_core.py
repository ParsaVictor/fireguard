# FireGuard — real-time fire & smoke intelligence on video
import os
import csv
import time
import math
import wave
import shutil
import warnings
import subprocess
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG — single place for every user-facing knob
# ============================================================

CONFIG = {
    # --- model ---
    "model": "yolo26s",          # default & recommended; alternates below stay available
    "imgsz": 640,
    "iou": 0.45,

    # --- operating profile (presets) ---
    # standard    : fire + smoke, balanced thresholds (default)
    # fire-only   : ignore smoke entirely (government / open-terrain mode)
    # early-smoke : ultra-sensitive smoke -> earliest possible warning
    "profile": "standard",

    # --- per-class confidence (overridden by profile unless set here) ---
    "conf_fire": None,           # None -> take from profile
    "conf_smoke": None,

    # --- speed ---
    "frame_skip": 1,             # run inference every Nth frame (1 = every frame)
    "start_frame": 0,            # begin analysis at this frame (skip intro)
    "max_frames": None,          # stop early (None = whole video)

    # --- robustness (temporal confirmation, kills flicker/false alarms) ---
    "confirm_window": 5,         # look at the last N frames
    "confirm_hits": 2,           # need >= K hits in that window to CONFIRM a class
    "clear_frames": 8,           # hold the alarm this many misses before clearing

    # --- alert tiers (visual + audio) ---
    # SMOKING  (gray frame)    : small smoke only, likely cigarette — no alarm sound
    # WARNING  (orange + beep) : large smoke, fire may be starting — beeper audio
    # DANGER/CRITICAL (red)    : fire confirmed — siren audio + pulsing red border
    "cigarette_max_area_frac": 0.004,   # smoke box smaller than 0.4% of frame -> SMOKING
    "audio_alerts": True,               # embed beep/siren audio into the output MP4
    "alert_cooldown_s": 4.0,            # repeat-alarm interval while hazard is active

    # --- output ---
    "output_dir": "outputs",
    "snapshot": True,            # save evidence PNG on every escalation
    "drive_backup": True,        # Colab: copy results to MyDrive/FireGuard_Outputs
    "display_samples": 6,        # annotated frames returned for inline display
}

PROFILES = {
    "standard":    {"fire": True, "smoke": True, "conf_fire": 0.30, "conf_smoke": 0.25},
    "fire-only":   {"fire": True, "smoke": False, "conf_fire": 0.30, "conf_smoke": 0.25},
    "early-smoke": {"fire": True, "smoke": True, "conf_fire": 0.35, "conf_smoke": 0.15},
}

# ============================================================
# MODEL REGISTRY — default is yolo26s; alternates kept for edge devices
# ============================================================

MODEL_REGISTRY = {
    "yolo26s": {
        "repo": "SalahALHaismawi/yolov26-fire-detection",
        "file": "best.pt",
        "params": "9.9M",
        "note": "DEFAULT — newest generation, most temporally stable, sees fire + smoke",
    },
    "yolov8n": {
        "repo": "rabahdev/fire-smoke-yolov8n",
        "file": "best.pt",
        "params": "3.0M",
        "note": "optional — fastest, for very weak CPUs / edge boxes",
    },
    "yolo11s": {
        "repo": "leeyunjai/yolo11-firedetect",
        "file": "firedetect-11s.pt",
        "params": "9.4M",
        "note": "optional — middle ground",
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_COLAB = False
try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass


def load_model(key: str, cache_dir: str = "models"):
    """Load a model from the registry (cached locally) or a custom .pt path."""
    from ultralytics import YOLO

    if key in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[key]
        local = os.path.join(cache_dir, f"{key}.pt")
        if not os.path.exists(local):
            os.makedirs(cache_dir, exist_ok=True)
            from huggingface_hub import hf_hub_download
            print(f"⬇️  Downloading {key} ({spec['repo']}) ...")
            path = hf_hub_download(repo_id=spec["repo"], filename=spec["file"])
            shutil.copy(path, local)
        weights = local
    elif os.path.isfile(key):
        weights = key
    else:
        raise FileNotFoundError(
            f"Unknown model '{key}'. Choose one of {list(MODEL_REGISTRY)} or pass a .pt path."
        )

    model = YOLO(weights)
    print(f"✅ Loaded '{os.path.basename(weights)}'  |  classes: {list(model.names.values())}  |  device: {DEVICE}")
    return model


# ============================================================
# DETECTION
# ============================================================

def normalize_class(name: str):
    """Map any dataset's label spelling to 'fire' / 'smoke' (or ignore)."""
    n = name.lower()
    if "fire" in n or "flame" in n:
        return "fire"
    if "smoke" in n:
        return "smoke"
    return None  # e.g. yolo26s extra class 'other'


@dataclass
class Detection:
    cls: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def centroid(self):
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def detect(model, frame, cfg) -> list:
    """Run inference + per-class confidence & size filtering."""
    conf_fire = cfg["_conf_fire"]
    conf_smoke = cfg["_conf_smoke"]
    min_area = 0.0008 * frame.shape[0] * frame.shape[1]

    results = model.predict(
        source=frame,
        imgsz=cfg["imgsz"],
        conf=min(conf_fire, conf_smoke),
        iou=cfg["iou"],
        device=DEVICE,
        verbose=False,
    )

    dets = []
    for box in results[0].boxes:
        cls = normalize_class(model.names[int(box.cls[0])])
        if cls is None or not cfg["_use"].get(cls, False):
            continue
        conf = float(box.conf[0])
        if conf < (conf_fire if cls == "fire" else conf_smoke):
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        d = Detection(cls, conf, x1, y1, x2, y2)
        if d.area < min_area:
            continue
        dets.append(d)
    return dets


# ============================================================
# HAZARD ENGINE — temporal confirmation + tiered alert state machine
# ============================================================

HAZARD_STATES = {
    "SAFE":     {"color": (89, 199, 52),   "label": "SAFE"},
    "CAUTION":  {"color": (10, 214, 255),  "label": "SIGHTING (unconfirmed)"},
    "SMOKING":  {"color": (170, 170, 170), "label": "SMOKING / CIGARETTE"},     # gray frame
    "WARNING":  {"color": (0, 165, 255),   "label": "EARLY FIRE — SMOKE"},      # orange + beeper
    "DANGER":   {"color": (48, 59, 255),   "label": "FIRE — SIREN"},            # red
    "CRITICAL": {"color": (48, 59, 255),   "label": "FIRE + SMOKE — SIREN"},    # red, stronger
}
_ORDER = ["SAFE", "CAUTION", "SMOKING", "WARNING", "DANGER", "CRITICAL"]


class HazardEngine:
    """Confirms classes over time so one flickering frame can't raise an alarm,
    and one missed frame can't silently clear one (hysteresis).
    Confirmed smoke is split into two tiers by box size:
    small -> SMOKING (cigarette, gray), large -> WARNING (early fire, orange)."""

    def __init__(self, cfg):
        self.w = cfg["confirm_window"]
        self.k = cfg["confirm_hits"]
        self.clear = cfg.get("clear_frames", 8)
        self.cig_frac = cfg.get("cigarette_max_area_frac", 0.004)
        self.hist = {"fire": deque(maxlen=self.w), "smoke": deque(maxlen=self.w)}
        self.miss = {"fire": self.clear, "smoke": self.clear}
        self.active = {"fire": False, "smoke": False}
        self.state = "SAFE"
        self.cooldown_s = cfg["alert_cooldown_s"]
        self._last_alarm_t = 0.0

    def update(self, dets, frame_area) -> dict:
        present = {"fire": any(d.cls == "fire" for d in dets),
                   "smoke": any(d.cls == "smoke" for d in dets)}
        for c, hit in present.items():
            self.hist[c].append(hit)
            if hit:
                self.miss[c] = 0
            else:
                self.miss[c] += 1
            if not self.active[c] and sum(self.hist[c]) >= self.k:
                self.active[c] = True                       # confirm
            elif self.active[c] and self.miss[c] >= self.clear:
                self.active[c] = False                      # clear (hysteresis)

        fire, smoke = self.active["fire"], self.active["smoke"]
        smoke_frac = max((d.area for d in dets if d.cls == "smoke"), default=0) / max(frame_area, 1)
        big_smoke = smoke and smoke_frac >= self.cig_frac

        if fire and smoke:
            state = "CRITICAL"
        elif fire:
            state = "DANGER"
        elif smoke and big_smoke:
            state = "WARNING"          # early fire — orange + beeper
        elif smoke:
            state = "SMOKING"          # cigarette — gray frame
        elif any(present.values()):
            state = "CAUTION"
        else:
            state = "SAFE"

        changed = state != self.state
        escalated = _ORDER.index(state) > _ORDER.index(self.state)
        self.state = state

        now = time.time()
        repeat = (
            state in ("DANGER", "CRITICAL")
            and not changed
            and now - self._last_alarm_t >= self.cooldown_s
        )
        if changed or repeat:
            self._last_alarm_t = now

        return {
            "state": state,
            "changed": changed,
            "escalated": escalated,
            "confirmed_fire": fire,
            "confirmed_smoke": smoke,
            "smoke_frac": smoke_frac,
        }


# ============================================================
# OVERLAY RENDERER — broadcast-style annotation
# ============================================================

FIRE_COLOR = (48, 59, 255)      # #FF3B30
SMOKE_COLOR = (0, 196, 255)     # #FFC400


def _blend(img, x1, y1, x2, y2, color, alpha):
    """Alpha-blend a solid color over a rectangular region, in place."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = (roi * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)


def _chip(img, text, x, y, color, scale=0.55, thick=2, pad=6, text_color=(255, 255, 255)):
    """Text on a filled label chip; returns bottom-left of the chip."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    h = th + 2 * pad
    y0 = y - h if y - h >= 0 else y          # flip below if clipped at top
    _blend(img, x, y0, x + tw + 2 * pad, y0 + h, color, 0.85)
    cv2.putText(img, text, (x + pad, y0 + h - pad - bl + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv2.LINE_AA)
    return y0


def _brackets(img, x1, y1, x2, y2, color, thick=3):
    """Professional corner brackets instead of a full rectangle."""
    l = int(np.clip(0.22 * min(x2 - x1, y2 - y1), 10, 36))
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * l, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * l), color, thick, cv2.LINE_AA)


def _dashed_circle(img, center, radius, color, n_seg=20, thick=2, alpha=0.0):
    step = int(360 / n_seg)
    overlay = img.copy() if alpha else img
    for a in range(0, 360, 2 * step):
        cv2.ellipse(overlay, center, (radius, radius), 0, a, a + step, color, thick, cv2.LINE_AA)
    if alpha:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_overlay(frame, dets, eng_info, stats, t_idx) -> np.ndarray:
    """All visual annotation: state styling, region marking, HUD, borders."""
    img = frame.copy()
    H, W = img.shape[:2]
    state = eng_info["state"]
    sc = HAZARD_STATES[state]["color"]

    # --- cigarette mode: grayscale frame, region still marked on top ---
    if state == "SMOKING":
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # --- per-detection annotation (region marking in EVERY state) ---
    for d in dets:
        color = FIRE_COLOR if d.cls == "fire" else SMOKE_COLOR
        if d.cls == "fire":
            _blend(img, d.x1, d.y1, d.x2, d.y2, color, 0.14)     # highlighted fire region
        _brackets(img, d.x1, d.y1, d.x2, d.y2, color)
        icon = "FIRE" if d.cls == "fire" else "SMOKE"
        _chip(img, f"{icon} {d.conf * 100:.1f}%", d.x1, d.y1 - 6, color)

        if d.cls == "fire":                                       # safety zone around the fire
            cx, cy = d.centroid
            r = int(0.65 * max(d.x2 - d.x1, d.y2 - d.y1))
            _dashed_circle(img, (cx, cy), r, FIRE_COLOR, thick=2, alpha=0.45)
            cv2.line(img, (cx - 10, cy), (cx + 10, cy), FIRE_COLOR, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy - 10), (cx, cy + 10), FIRE_COLOR, 2, cv2.LINE_AA)

    # --- alert borders: red pulsing for fire, orange pulsing for early smoke,
    #     static gray frame border for cigarette mode ---
    if state in ("DANGER", "CRITICAL"):
        a = 0.30 + 0.22 * math.sin(2 * math.pi * t_idx / 15.0)
        b = 8 if state == "CRITICAL" else 5
        _blend(img, 0, 0, W, b, sc, a)
        _blend(img, 0, H - b, W, H, sc, a)
        _blend(img, 0, 0, b, H, sc, a)
        _blend(img, W - b, 0, W, H, sc, a)
    elif state == "WARNING":
        a = 0.30 + 0.18 * math.sin(2 * math.pi * t_idx / 10.0)
        _blend(img, 0, 0, W, 5, sc, a)
        _blend(img, 0, H - 5, W, H, sc, a)
        _blend(img, 0, 0, 5, H, sc, a)
        _blend(img, W - 5, 0, W, H, sc, a)
    elif state == "SMOKING":
        _blend(img, 0, 0, W, 3, sc, 0.45)
        _blend(img, 0, H - 3, W, H, sc, 0.45)
        _blend(img, 0, 0, 3, H, sc, 0.45)
        _blend(img, W - 3, 0, W, H, sc, 0.45)

    # --- HUD (top-left, always on top) ---
    hud1 = f"FireGuard  |  {HAZARD_STATES[state]['label']}"
    hud2 = (f"{stats['fps']:.1f} FPS  |  fire:{stats['n_fire']}  smoke:{stats['n_smoke']}"
            f"  |  frame {stats['frame']}")
    scale = 0.55 if H >= 480 else 0.45
    y0 = _chip(img, hud1, 10, 12, sc, scale=scale)
    _chip(img, hud2, 10, y0 + 4, (20, 20, 20), scale=scale * 0.92, thick=1,
          text_color=(240, 240, 240), pad=5)
    return img


# ============================================================
# ALERT AUDIO — beeper (early smoke) / siren (fire) embedded in the MP4
# ============================================================

def _build_alert_audio(timeline, fps, path, sr=22050):
    """Synthesize a mono WAV with beep patterns for WARNING and DANGER/CRITICAL
    stretches. Returns True if the track has any sound."""
    try:
        states = [e["state"] for e in timeline]
        if not states:
            return False
        dur = len(states) / fps + 0.5
        n = int(dur * sr)
        t = np.arange(n) / sr

        def sample_mask(frame_mask):
            reps = int(np.ceil(sr / fps))
            m = np.repeat(frame_mask, reps)
            return m[:n] if len(m) >= n else np.pad(m, (0, n - len(m)))

        warn = sample_mask(np.array([s == "WARNING" for s in states]))
        fire = sample_mask(np.array([s in ("DANGER", "CRITICAL") for s in states]))
        audio = np.zeros(n)
        if warn.any():                                   # 2 Hz beeps @ 880 Hz
            audio += warn * (np.sin(2 * np.pi * 2.0 * t) > 0) * 0.55 * np.sin(2 * np.pi * 880 * t)
        if fire.any():                                   # urgent 4 Hz siren @ 1200 Hz
            audio += fire * (np.sin(2 * np.pi * 4.0 * t) > 0) * 0.8 * np.sin(2 * np.pi * 1200 * t)
        if not audio.any():
            return False

        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        return True
    except Exception as e:                                # audio is best-effort
        print(f"   ⚠️ audio track skipped: {e}")
        return False


def _mux_audio(video, wav, out):
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-i", wav,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
         "-shortest", out],
        capture_output=True,
    )
    return r.returncode == 0


def _drive_backup(cfg):
    """Colab only: copy results to MyDrive/FireGuard_Outputs."""
    if not (IN_COLAB and cfg.get("drive_backup", True)):
        return None
    src_dir = cfg["output_dir"]
    dest = "/content/drive/MyDrive/FireGuard_Outputs"
    try:
        os.makedirs(dest, exist_ok=True)
        for f in os.listdir(src_dir):
            p = os.path.join(src_dir, f)
            if os.path.isfile(p):
                shutil.copy2(p, dest)
        snaps = os.path.join(src_dir, "snapshots")
        if os.path.isdir(snaps):
            shutil.copytree(snaps, os.path.join(dest, "snapshots"), dirs_exist_ok=True)
        return dest
    except Exception as e:
        print(f"   ⚠️ Drive backup failed: {e}")
        return None


# ============================================================
# PIPELINE
# ============================================================

def _resolve_profile(cfg):
    prof = PROFILES[cfg["profile"]]
    cfg["_use"] = {"fire": prof["fire"], "smoke": prof["smoke"]}
    cfg["_conf_fire"] = cfg["conf_fire"] if cfg["conf_fire"] is not None else prof["conf_fire"]
    cfg["_conf_smoke"] = cfg["conf_smoke"] if cfg["conf_smoke"] is not None else prof["conf_smoke"]
    return cfg


def _fmt_eta(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def process_video(model, cfg, src, out_name="result.mp4", max_frames=None, quiet=False):
    """Main loop: read -> detect -> confirm -> annotate -> write + log + audio."""
    cfg = _resolve_profile(dict(cfg))
    max_frames = max_frames or cfg["max_frames"]

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {src}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(cfg["output_dir"], exist_ok=True)
    snap_dir = os.path.join(cfg["output_dir"], "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    temp = os.path.join(cfg["output_dir"], "_temp.mp4")
    final = os.path.join(cfg["output_dir"], out_name)
    writer = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    eng = HazardEngine(cfg)
    events, samples = [], []
    for _ in range(cfg.get("start_frame", 0)):   # fast-forward without decoding
        if not cap.grab():
            break
    n_total = min(total - cfg.get("start_frame", 0), max_frames) if max_frames else total - cfg.get("start_frame", 0)
    sample_at = set(np.linspace(0, max(n_total - 1, 1), cfg["display_samples"]).astype(int)) if n_total > 1 else {0}
    print_every = max(n_total // 25, 1)

    if not quiet:
        est = n_total * 0.15 if DEVICE == "cpu" else n_total * 0.02
        print(f"🎥 {W}x{H} @ {fps:.0f}fps — {n_total} frames to process"
              f"  (rough ETA on {DEVICE.upper()}: ~{_fmt_eta(est)})")

    stats = {"fps": 0.0, "n_fire": 0, "n_smoke": 0, "frame": 0}
    t0, frames_done, det_frames, det_time = time.time(), 0, 0, 0.0
    fire_ever, smoke_ever = False, False
    timeline = []               # per-frame state history for the summary plot + audio
    last_dets = []
    last_info = {"state": "SAFE", "changed": False, "escalated": False,
                 "confirmed_fire": False, "confirmed_smoke": False, "smoke_frac": 0.0}
    emap = {"SAFE": "🟢", "CAUTION": "🟡", "SMOKING": "🚬", "WARNING": "🟠",
            "DANGER": "🚨", "CRITICAL": "🚨🚨"}

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frames_done >= max_frames):
            break
        idx = frames_done
        frames_done += 1

        if idx % cfg["frame_skip"] == 0:
            ti = time.time()
            dets = detect(model, frame, cfg)
            det_time += time.time() - ti
            det_frames += 1
            last_dets = dets
        else:
            dets = last_dets  # reuse boxes between skipped inferences

        info = eng.update(dets, W * H)
        stats.update(n_fire=sum(d.cls == "fire" for d in dets),
                     n_smoke=sum(d.cls == "smoke" for d in dets), frame=idx + 1)
        stats["fps"] = (idx + 1) / max(time.time() - t0, 1e-6)

        annotated = draw_overlay(frame, dets, info, stats, t_idx=idx)

        # --- event logging on every state change ---
        if info["changed"]:
            msg = f"{emap[info['state']]} frame {idx + 1}: state -> {info['state']}"
            if not quiet:
                print(msg)
            events.append({"frame": idx + 1, "time_s": round(idx / fps, 2),
                           "state": info["state"],
                           "fire": stats["n_fire"], "smoke": stats["n_smoke"]})
            if cfg["snapshot"] and info["escalated"]:
                cv2.imwrite(os.path.join(snap_dir, f"{info['state']}_{idx + 1:06d}.png"), annotated)
        fire_ever |= info["confirmed_fire"]
        smoke_ever |= info["confirmed_smoke"]
        timeline.append({"frame": idx + 1, "time_s": round(idx / fps, 2),
                         "state": info["state"], "n_fire": stats["n_fire"],
                         "n_smoke": stats["n_smoke"]})

        if idx in sample_at:
            samples.append(annotated)
        writer.write(annotated)

        # --- progress with ETA (visible end in sight) ---
        if not quiet and ((idx + 1) % print_every == 0 or idx + 1 == n_total):
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n_total - idx - 1)
            print(f"   ▏{100 * (idx + 1) / n_total:5.1f}%  ({idx + 1}/{n_total})  "
                  f"{stats['fps']:.1f} FPS  |  ETA {_fmt_eta(eta)}  |  {emap[info['state']]} {info['state']}")

    cap.release()
    writer.release()

    # events CSV
    if events:
        with open(os.path.join(cfg["output_dir"], "events.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "time_s", "state", "fire", "smoke"])
            w.writeheader()
            w.writerows(events)

    # encode to H.264, then mux the beep/siren track (best effort)
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-i", temp, "-vcodec", "libx264", "-crf", "23", final],
                           capture_output=True)
        if r.returncode == 0:
            os.remove(temp)
        else:
            shutil.move(temp, final)
        if cfg.get("audio_alerts", True) and timeline:
            wav = os.path.join(cfg["output_dir"], "_alerts.wav")
            if _build_alert_audio(timeline, fps, wav):
                with_audio = final.replace(".mp4", "_audio.mp4")
                if _mux_audio(final, wav, with_audio):
                    os.replace(with_audio, final)
                os.remove(wav)
    else:
        shutil.move(temp, final)

    drive_dest = _drive_backup(cfg)

    summary = {
        "frames": frames_done,
        "inferences": det_frames,
        "ms_per_inference": 1000 * det_time / max(det_frames, 1),
        "pipeline_fps": frames_done / max(time.time() - t0, 1e-6),
        "events": len(events),
        "fire_seen": fire_ever,
        "smoke_seen": smoke_ever,
        "final_state": eng.state,
        "output": final,
        "drive": drive_dest,
        "timeline": timeline,
    }
    if not quiet:
        print("\n📊 SUMMARY")
        for k, v in summary.items():
            if k != "timeline":
                print(f"   {k:>18}: {v}")
        if drive_dest:
            print(f"\n☁️  Results copied to Google Drive: {drive_dest}")
    return summary, samples


if __name__ == "__main__":
    import sys
    model_key = sys.argv[1] if len(sys.argv) > 1 else "yolo26s"
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    model = load_model(model_key)
    cfg = dict(CONFIG)
    cfg["display_samples"] = 5
    cfg["start_frame"] = 3400
    summary, samples = process_video(model, cfg, "fire_detection_result.mp4",
                                     out_name=f"result_{model_key}.mp4", max_frames=n_frames)
    for i, s in enumerate(samples[:5]):
        cv2.imwrite(f"sample_{model_key}_{i}.png", s)
    print("samples saved:", len(samples))
