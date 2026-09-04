# 🔥 FireGuard

**Fire & smoke detection for indoor CCTV — one notebook from raw datasets to a deployable model.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ParsaVictor/fireguard/blob/main/FireGuard_Pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/detector-YOLO26-8A2BE2)
![Status](https://img.shields.io/badge/status-v0.2%20training%20pipeline-orange)

> **Status — v0.2.** The training pipeline is complete and tested; **accuracy numbers are not filled in yet**
> (see [Results](#results)). v0.1 shipped an inference demo on third-party checkpoints of unknown
> provenance — v0.2 replaces those with a model trained on documented, licensed data.
>
> 🇮🇷 [خلاصهٔ فارسی](#فارسی) در انتهای فایل.

---

## Scope

FireGuard targets **indoor and urban CCTV** — homes, offices, shops. Not wildfire towers, not drones.

That choice drives everything else: which datasets are useful, which augmentations matter, and why
the model is evaluated on false-alarm rate rather than mAP alone.

| | |
|---|---|
| **Classes** | `0 = fire`, `1 = smoke` — two, nothing else |
| **Input** | wide-angle IP cameras, 320×240 → 720p, heavy compression, IR night mode |
| **Detector** | YOLO26 (`n`/`s`/`m`), fine-tuned from COCO weights |
| **Metric that matters** | false alarms per camera per 24 h, then time-to-detection |

---

## Quickstart

**[▶ Open `FireGuard_Pipeline.ipynb` in Colab](https://colab.research.google.com/github/ParsaVictor/fireguard/blob/main/FireGuard_Pipeline.ipynb)**
→ `Runtime → Change runtime type → T4 GPU` → run the cells in order.

One notebook, 14 sections, dataset download through TensorRT export:

| § | Step | Every session? |
|---|---|---|
| 1–2 | Install, hardware auto-tune, settings | ✅ |
| 3 | Parallel dataset download → Drive | first run |
| 4–5 | Merge, class fix, dedup, group-aware split, audit | first run |
| 6–8 | CCTV augmentation, 🚁 3-minute smoke test, baseline | first run |
| 9 | 🔬 A/B ablation on augmentation | optional |
| 10 | **Training** — progressive resolution, auto-resume | ✅ |
| 11–14 | Evaluation, operating point, export, model card | after training |

Weights land in `MyDrive/FireGuard_Runs/<RUN>/final/` with a `model_card.json`.

Do not hand-edit the notebook — edit `build_pipeline_notebook.py` and regenerate:

```bash
python build_pipeline_notebook.py
```

---

## Data

| Dataset | Images | Composition | Format | License |
|---|---|---|---|---|
| **FASDD_CV** | 95,126 | 39,114 negative · 23,350 smoke · 20,126 both · 12,536 fire | YOLO / VOC / COCO | **CC BY 4.0** |
| **D-Fire** | 21,527 | 9,838 negative · 5,867 smoke · 4,658 both · 1,164 fire | YOLO | free |
| **Merged** | ~113k | ~37k with fire · ~53k with smoke · ~48k negative | YOLO | — |

FASDD_CV is the backbone: it explicitly spans indoor/outdoor, day/night, near/far, and surveillance
cameras. D-Fire contributes its 9,838 deliberately-confusing negatives — sunsets, lamp glare,
cloud-that-looks-like-smoke.

Deliberately **excluded**: Pyro-SDIS, FIgLib, FLAME (wildfire towers and drones — wrong domain) and
DetectiumFire (CC BY-NC, unusable in a product).

Numbers above were measured directly from the archives, not quoted from papers — see
[`DATASET_AUDIT.md`](DATASET_AUDIT.md).

---

## Traps this pipeline handles

Each of these was measured, not assumed. Any of them silently ruins a naive merge.

| # | Trap | Handling |
|---|---|---|
| 1 | **Class maps are opposite.** FASDD is `0=fire`, D-Fire is `0=smoke` | D-Fire labels flipped with `1-c` |
| 2 | **D-Fire's `AoF` split is consecutive video frames** — adjacent frames measured 80–90 % similar | group-aware split |
| 3 | Those same-event frames sit **6–13 bits apart** — far outside the dedup threshold of 3 | second threshold: distance 4–12 → keep both, lock to one split |
| 4 | ~39k plain negatives (sky, walls) share **identical hashes** and formed a bucket the pairwise search skipped | exact-hash pass in O(n) before near-duplicate search |
| 5 | **`imgsz > 640` is wasted** — FASDD images are capped at 640 px on the long side | hard ceiling at 640 |
| 6 | **`resume=True` overwrites every arg from the checkpoint** (`trainer.py`: `self.args = get_cfg(ckpt_args)`) | each resolution stage is a fresh `train()`; resume only *within* a stage |
| 7 | Google Drive is very slow with many small files | data and weights on local disk; Drive sync on a background thread |
| 8 | Public datasets are clean web images; real CCTV is not | compression / downscale / grayscale / motion-blur augmentation via Ultralytics' official `augmentations=` hook |

A known label-semantics caveat: FASDD labels **candles and matches as `fire`**. Left in for now —
the plan is to separate them in the decision layer by size and persistence rather than by deleting data.

---

## Results

Not yet measured. Filled in after the first full training run.

| Metric | Value |
|---|---|
| mAP@50 — overall | — |
| mAP@50 — fire | — |
| mAP@50 — smoke | — |
| False alarms / camera / 24 h | — |
| Latency (T4, TensorRT FP16) | — |

Note: smoke mAP is always lower than fire mAP. Smoke has no crisp boundary — two expert annotators
will not draw the same box. Report them separately.

---

## Where this is going

Three tiers sharing one evidence bus, detailed in [`PLAN_V2.md`](PLAN_V2.md):

- **SPARK** — edge tier (RPi5 / Jetson Orin Nano), motion-gated inference
- **BLAZE** — commercial tier: tracking + a flame-flicker DFT verifier (real flames oscillate at 2–12 Hz;
  sunsets and traffic cones do not) + plume-growth verification, fused as log-odds per tracked fire
- **INFERNO** — server tier: multi-resolution ensemble, VLM adjudication, abstention, and distillation
  back into the smaller two

---

## Repository

```
FireGuard_Pipeline.ipynb      ← the notebook: data → training → export
build_pipeline_notebook.py    ← its generator (edit this, not the notebook)
fireguard_core.py             ← v0.1 inference engine: hazard state machine + overlays
FireGuard.ipynb               ← v0.1 inference demo
PLAN_V2.md                    ← three-tier architecture
DATA_AND_MODELS.md            ← model and dataset selection, with reasoning
DATASET_AUDIT.md              ← measured dataset facts and the eight traps
legacy/                       ← superseded two-notebook version
```

---

## License

[MIT](LICENSE) for this code. Datasets keep their own licenses (FASDD_CV is CC BY 4.0 — attribution required).

⚠️ The current pipeline builds on Ultralytics YOLO, which is **AGPL-3.0**. For a closed commercial
product the same recipe should be ported to D-FINE or RF-DETR (both Apache-2.0). See `DATA_AND_MODELS.md`.

> FireGuard is an assistive tool. Follow local fire-safety regulations; never deploy it as the sole
> life-safety system.

---

## فارسی

**FireGuard** یک سامانهٔ تشخیص آتش و دود برای **دوربین مداربستهٔ داخلی** است — خانه، دفتر، مغازه.
نه برج جنگلی، نه پهپاد.

- **دو کلاس و بس:** `0 = fire` · `1 = smoke`
- **یک نوت‌بوک:** [`FireGuard_Pipeline.ipynb`](FireGuard_Pipeline.ipynb) — از دانلود دیتاست تا خروجی TensorRT
- **داده:** FASDD_CV (۹۵,۱۲۶ تصویر، CC BY 4.0) + D-Fire (۲۱,۵۲۷) ≈ **۱۱۳ هزار تصویر**
- **هشت تله** که همه‌شان اندازه‌گیری شدند و در خط لوله حل شده‌اند — مهم‌ترینشان اینکه
  **نگاشت کلاس دو دیتاست دقیقاً برعکس هم است**
- **معیاری که مهم است:** نرخ آلارم کاذب در هر دوربین در ۲۴ ساعت، نه فقط mAP

وضعیت: خط لولهٔ آموزش کامل و آزمایش‌شده است؛ **اعداد دقت هنوز پر نشده‌اند.**

نوت‌بوک را دستی ویرایش نکن — `build_pipeline_notebook.py` را عوض کن و دوباره بساز.
