# -*- coding: utf-8 -*-
"""سازندهٔ نوت‌بوک آموزش FireGuard.
نوت‌بوک را دستی ویرایش نکن — این فایل را عوض کن و دوباره اجرا کن:
    py -3 build_train_notebook.py
"""
import json
import os

NB = "FireGuard_Train.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


cells = []

# ============================================================ 0 · عنوان
cells.append(md(r"""
# 🔥 FireGuard — آموزش مدل

آموزش یک آشکارساز آتش و دود روی ~۱۱۳ هزار تصویر، مخصوصِ **دوربین مداربستهٔ داخلی**.

پیش‌نیاز: نوت‌بوک `FireGuard_Dataset.ipynb` را اجرا کرده باشی تا
`fireguard_data.zip` در درایو ساخته شده باشد.

---

### نقشهٔ نوت‌بوک

| بخش | چه می‌کند | هر جلسه؟ |
|---|---|---|
| ۱ | نصب و بررسی GPU | ✅ |
| ۲ | آوردن داده از درایو به دیسک محلی | ✅ |
| ۳ | **بازرسی داده پیش از آموزش** | بار اول |
| ۴ | خط پایه — سنجش مدل خام YOLO26 | بار اول |
| ۵ | **آموزش** (با resume خودکار) | ✅ |
| ۶ | ارزیابی — mAP هر کلاس، PR، ماتریس اشتباه | بعد از آموزش |
| ۷ | **انتخاب نقطهٔ کار بر اساس آلارم کاذب** | بعد از آموزش |
| ۸ | خروجی ONNX/TensorRT + سنجش سرعت واقعی | بعد از آموزش |

---

### ⚠️ سه قانون که این نوت‌بوک رعایت می‌کند

| قانون | چرا |
|---|---|
| **داده روی `/content` می‌ماند، نه درایو** | خواندن ۱۱۳ هزار فایل کوچک از درایو (FUSE) آموزش را ۵ تا ۲۰ برابر کند می‌کند |
| **وزن‌ها هم روی `/content` نوشته می‌شوند، بعد به درایو کپی می‌شوند** | نوشتن مکرر checkpoint روی FUSE هم کند است و هم گاهی فایل را خراب می‌کند |
| **`resume` خودکار** | Colab رایگان قطع می‌شود؛ سلول ۵ خودش `last.pt` را از درایو برمی‌گرداند |
"""))

# ============================================================ 1 · نصب
cells.append(md(r"""
## ۱ · نصب و بررسی سخت‌افزار
"""))

cells.append(code(r"""
# --- بستهٔ اصلی ---
!pip install -q --upgrade ultralytics

# --- augmentation پیشرفته (برای پرکردن شکاف CCTV) ---
!pip install -q albumentations

# --- ابزار ارزیابی و رسم ---
!pip install -q "matplotlib>=3.8" pandas seaborn

# --- خروجی صنعتی (اختیاری ولی توصیه‌شده) ---
!pip install -q onnx onnxruntime-gpu onnxslim

import torch, ultralytics, platform, os, subprocess
print('ultralytics :', ultralytics.__version__)
print('torch       :', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU         : {p.name}  |  {p.total_memory/1e9:.1f} GB')
else:
    print('⚠️ GPU روشن نیست:  Runtime → Change runtime type → T4 GPU')
print('RAM         :', round(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9,1), 'GB')
!df -h /content | tail -1
"""))

# ============================================================ 2 · تنظیمات
cells.append(md(r"""
## ۲ · تنظیمات و آوردن داده

**تنها سلولی که دست می‌زنی.** بقیه را فقط اجرا کن.
"""))

cells.append(code(r"""
from google.colab import drive
drive.mount('/content/drive')

import os, shutil, time, zipfile

# ══════════════════ تنظیمات ══════════════════
DRIVE_DIR  = '/content/drive/MyDrive/FireGuard_Datasets'   # جایی که دیتاست هست
RUNS_DRIVE = '/content/drive/MyDrive/FireGuard_Runs'       # جایی که وزن‌ها ذخیره می‌شوند

MODEL   = 'yolo26s.pt'   # yolo26n=سریع‌ترین · yolo26s=تعادل (پیشنهاد) · yolo26m=دقیق‌تر
RUN     = 'fireguard_s_v1'   # نام اجرا — برای شروع آزمایش جدید عوضش کن
EPOCHS  = 120
IMGSZ   = 640            # بالاتر نبر: تصاویر FASDD خودشان حداکثر ۶۴۰ پیکسل‌اند
BATCH   = 16             # T4: n→32 · s→16 · m→8   (اگر OOM داد نصف کن)

SYNC_EVERY = 5           # هر چند epoch وزن‌ها به درایو کپی شوند
# ═════════════════════════════════════════════

DATA_LOCAL = '/content/fireguard_data'
RUNS_LOCAL = '/content/runs'
os.makedirs(RUNS_DRIVE, exist_ok=True)
os.makedirs(RUNS_LOCAL, exist_ok=True)

# ---------- آوردن دیتاست از درایو به دیسک محلی ----------
if os.path.isdir(f'{DATA_LOCAL}/images/train'):
    print('✅ دیتاست از قبل روی دیسک محلی هست')
else:
    z = os.path.join(DRIVE_DIR, 'fireguard_data.zip')
    if not os.path.exists(z):
        raise FileNotFoundError(
            'fireguard_data.zip پیدا نشد.\n'
            'اول نوت‌بوک FireGuard_Dataset.ipynb را اجرا کن.')
    print(f'📦 باز کردن {os.path.getsize(z)/1e9:.1f} گیگ روی دیسک محلی ...')
    t = time.time()
    os.makedirs(DATA_LOCAL, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA_LOCAL)
    print(f'   ✅ {time.time()-t:.0f} ثانیه')

# ---------- fire.yaml را به مسیر محلی سنجاق کن ----------
YAML = f'{DATA_LOCAL}/fire.yaml'
with open(YAML, 'w') as f:
    f.write('\n'.join([
        f'path: {DATA_LOCAL}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        '', 'nc: 2', 'names:', '  0: fire', '  1: smoke', '']))

for s in ['train', 'val', 'test']:
    n = len(os.listdir(f'{DATA_LOCAL}/images/{s}'))
    print(f'   {s:5}: {n:>7,} تصویر')
print('\nfire.yaml →', YAML)
"""))

# ============================================================ 3 · بازرسی
cells.append(md(r"""
## ۳ · بازرسی داده — **پیش از سوزاندن ساعت‌ها GPU**

سه چیز را می‌سنجد که اگر خراب باشند کل آموزش هدر می‌رود:

1. **نگاشت کلاس** — آیا `0` واقعاً آتش است؟ (چشمی)
2. **هندسهٔ برچسب** — مختصات بیرون از بازه، کادر صفر، کلاس نامعتبر
3. **توازن** — نسبت آتش/دود/منفی در هر split
"""))

cells.append(code(r"""
import glob, os, random, collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ---------- ۱) اعتبار هندسی برچسب‌ها ----------
issues = collections.Counter()
areas, per_split = [], collections.defaultdict(collections.Counter)

for s in ['train', 'val', 'test']:
    for lp in glob.glob(f'{DATA_LOCAL}/labels/{s}/*.txt'):
        lines = [l for l in open(lp).read().splitlines() if l.strip()]
        if not lines:
            per_split[s]['negative'] += 1
            per_split[s]['images'] += 1
            continue
        per_split[s]['images'] += 1
        cls_here = set()
        for l in lines:
            t = l.split()
            if len(t) != 5: issues['بد-فرمت'] += 1; continue
            c = int(float(t[0])); x, y, w, h = map(float, t[1:])
            if c not in (0, 1):                       issues['کلاس نامعتبر'] += 1
            if w <= 0 or h <= 0:                      issues['کادر صفر'] += 1
            if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                                                      issues['بیرون از کادر'] += 1
            if w*IMGSZ < 2 or h*IMGSZ < 2:            issues['خیلی نازک'] += 1
            areas.append(w*h); cls_here.add(c)
            per_split[s]['box_fire' if c == 0 else 'box_smoke'] += 1
        if 0 in cls_here: per_split[s]['img_fire'] += 1
        if 1 in cls_here: per_split[s]['img_smoke'] += 1

print('=== اعتبار برچسب‌ها ===')
print('  ✅ هیچ مشکلی نیست' if not issues else f'  ⚠️ {dict(issues)}')

# ---------- ۲) توازن ----------
print('\n=== توازن ===')
print(f"{'split':6}{'تصویر':>9}{'آتش‌دار':>10}{'دوددار':>10}{'منفی':>9}{'کادر آتش':>11}{'کادر دود':>11}")
print('-'*66)
for s in ['train', 'val', 'test']:
    d = per_split[s]
    print(f"{s:6}{d['images']:>9,}{d['img_fire']:>10,}{d['img_smoke']:>10,}"
          f"{d['negative']:>9,}{d['box_fire']:>11,}{d['box_smoke']:>11,}")

a = np.array(areas)
small = (a*IMGSZ*IMGSZ < 32*32).mean()
print(f"\nمساحت کادر (نسبت به فریم): میانه {np.median(a):.4f}  |  p10 {np.percentile(a,10):.4f}")
print(f"سهم اشیای «کوچک» (زیر ۳۲×۳۲ پیکسل در {IMGSZ}): {small*100:.1f}%")
print('→ همین عدد است که تشخیص زودهنگام دود را تعیین می‌کند؛ imgsz را پایین‌تر نبر.')

# ---------- ۳) بازبینی چشمی نگاشت کلاس ----------
random.seed(0)
files = random.sample(glob.glob(f'{DATA_LOCAL}/images/train/*'), 8)
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
COL = {0: ('#FF3B30', 'fire'), 1: ('#00A8FF', 'smoke')}
for ax, ip in zip(axes.ravel(), files):
    im = Image.open(ip); W, H = im.size
    ax.imshow(im); ax.axis('off')
    lp = f"{DATA_LOCAL}/labels/train/{os.path.splitext(os.path.basename(ip))[0]}.txt"
    for l in open(lp).read().strip().splitlines():
        t = l.split()
        if len(t) != 5: continue
        c = int(t[0]); cx, cy, w, h = map(float, t[1:])
        col, nm = COL[c]
        ax.add_patch(patches.Rectangle(((cx-w/2)*W, (cy-h/2)*H), w*W, h*H,
                                       fill=False, edgecolor=col, lw=2.5))
        ax.text((cx-w/2)*W, (cy-h/2)*H-4, nm, color=col, fontsize=9, weight='bold')
plt.suptitle('🔴 قرمز = 0 fire      🔵 آبی = 1 smoke      ← اگر جابه‌جاست، آموزش را شروع نکن',
             fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ============================================================ 4 · خط پایه
cells.append(md(r"""
## ۴ · خط پایه — مدل خام YOLO26

قبل از آموزش، عددِ شروع را ثبت می‌کنیم تا بعداً بتوانیم بگوییم «چقدر بهتر شدیم».

**انتظار: صفر.** مدل خام روی COCO آموزش دیده و در ۸۰ کلاسش نه `fire` هست نه `smoke` —
پس اصلاً نمی‌تواند آتش را تشخیص دهد. این سلول همین را ثابت می‌کند، و همان
«بهتر از YOLO26 خام» را به یک ادعای اندازه‌گیری‌شده تبدیل می‌کند.
"""))

cells.append(code(r"""
from ultralytics import YOLO
import time, torch

base = YOLO(MODEL)
print('کلاس‌های مدل خام:', len(base.names), 'کلاس')
has_fire = [n for n in base.names.values() if 'fire' in n.lower() or 'smoke' in n.lower()]
print("کلاس مرتبط با آتش/دود در COCO:", has_fire if has_fire else '❌ هیچ‌کدام')

# سرعت خام برای مقایسهٔ بعدی
dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
_ = base.predict(dummy, imgsz=IMGSZ, verbose=False)          # warm-up
t = time.time()
for _ in range(30):
    _ = base.predict(dummy, imgsz=IMGSZ, verbose=False)
BASE_MS = (time.time()-t)/30*1000
print(f'\nتأخیر مدل خام ({MODEL}, PyTorch, imgsz={IMGSZ}): {BASE_MS:.1f} ms/frame')
print('→ این عدد را نگه می‌داریم؛ در بخش ۸ با مدل نهایی مقایسه می‌شود.')
"""))

# ============================================================ 5 · آموزش
cells.append(md(r"""
## ۵ · آموزش

### چرا این تنظیمات، و نه پیش‌فرض‌ها

| تنظیم | مقدار | دلیل |
|---|---|---|
| `hsv_h` | **0.015** (خیلی کم) | رنگ نارنجی-قرمز **نصف سیگنال آتش** است. جابه‌جایی زیادِ رنگ به شبکه یاد می‌دهد رنگ را نادیده بگیرد. |
| `hsv_v` | **0.5** (زیاد) | دوربین مداربسته در نور کم و شب کار می‌کند |
| `degrees` | 5 | دوربین سقفی همیشه کمی کج است |
| `scale` | 0.5 | آتشِ نزدیک و دور، هر دو |
| `close_mosaic` | 15 | mosaic برای شیء بزرگ و بی‌شکل مثل ستون دود مضر است — ۱۵ epoch آخر خاموش می‌شود تا مدل روی فریم کامل تثبیت شود |
| `erasing` | 0.2 | انسداد جزئی (مبل جلوی آتش) |
| `freeze` | **ندارد** | وسوسه‌کننده است ولی بافت شعله آن‌قدر از اشیای COCO دور است که backbone باید تطبیق پیدا کند — فریزکردن ۳ تا ۵ واحد mAP می‌سوزاند |
| `cache` | **False** | ۱۱۳ هزار تصویر در RAM جا نمی‌شود (T4 فقط ~۱۲.۷ گیگ RAM دارد) |
| `workers` | 2 | بیشتر از ۲ روی Colab معمولاً گیر می‌کند |

### ★ و یک چیز که پیش‌فرض ندارد — پرکردن شکاف CCTV

دیتاست‌های عمومی تصویر **تمیزِ اینترنتی**اند. دوربین تو تصویر **کثیف** می‌دهد:
فشرده‌سازی سنگین، رزولوشن پایین، و شب‌ها خاکستریِ مادون‌قرمز.

سلول زیر سه augmentation اضافه می‌کند که Ultralytics به‌طور پیش‌فرض ندارد:

| افزوده | چه شکافی را می‌بندد |
|---|---|
| `ImageCompression(30..80)` | فشرده‌سازی JPEG دوربین |
| `Downscale(0.35..0.85)` | دوربین ۳۲۰×۲۴۰ و ۷۲۰p |
| `ToGray(p=0.12)` | **حالت شبانهٔ مادون‌قرمز** |
| `MotionBlur` | لرزش و حرکت |

بدون اینها مدل روی تصویر تمیز عالی و روی CCTV واقعی ضعیف می‌شود.

این از **قلّاب رسمیِ Ultralytics** استفاده می‌کند (`augmentations=`)، نه دست‌کاری داخلی
کتابخانه — پس با به‌روزرسانی نسخه نمی‌شکند.
"""))

cells.append(code(r"""
# ═══ augmentation مخصوص CCTV ═══
# از قلّابِ رسمیِ Ultralytics استفاده می‌کنیم، نه دست‌کاری داخلی کتابخانه.
# در augment.py خطِ ساختِ Albumentations این است:
#     Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None), ...)
# و در cfg/__init__.py کلیدِ "augmentations" صراحتاً مجاز شمرده شده.
# پس کافی است لیست را به model.train(augmentations=...) بدهیم.
#
# ⚠️ توجه: دادنِ این لیست، مجموعهٔ پیش‌فرض را کاملاً *جایگزین* می‌کند،
#    پس موارد مفیدِ پیش‌فرض را هم خودمان اینجا می‌آوریم.

CCTV_AUGS = None
try:
    import albumentations as A

    def _downscale():
        # نام پارامترها بین نسخه‌های albumentations فرق کرده
        try:    return A.Downscale(scale_range=(0.35, 0.85), p=0.20)
        except TypeError:
            return A.Downscale(scale_min=0.35, scale_max=0.85, p=0.20)

    CCTV_AUGS = [
        A.ImageCompression(quality_range=(30, 80), p=0.30),   # فشرده‌سازی سنگین دوربین
        _downscale(),                                          # ۳۲۰×۲۴۰ تا ۷۲۰p
        A.ToGray(p=0.12),                                      # ★ حالت شبانهٔ مادون‌قرمز
        A.MotionBlur(blur_limit=5, p=0.10),                    # لرزش و حرکت
        A.RandomBrightnessContrast(brightness_limit=0.25,
                                   contrast_limit=0.25, p=0.20),
        A.CLAHE(p=0.02),
    ]
    print('✅ augmentation مخصوص CCTV آماده شد:')
    for t in CCTV_AUGS:
        print(f'   · {t.__class__.__name__:26} p={getattr(t, "p", "?")}')
    print('\n   همه ImageOnly هستند → کادرها دست‌نخورده می‌مانند.')
except Exception as e:
    print(f'⚠️ albumentations در دسترس نیست ({e}) — با augmentation استاندارد ادامه می‌دهیم')
"""))

cells.append(code(r"""
import os, glob, shutil, torch
from ultralytics import YOLO

DRIVE_RUN  = f'{RUNS_DRIVE}/{RUN}'
LOCAL_RUN  = f'{RUNS_LOCAL}/{RUN}'
os.makedirs(DRIVE_RUN, exist_ok=True)

# ---------- resume: اگر جلسهٔ قبل قطع شده، وزن را از درایو برگردان ----------
resume = False
drive_last = f'{DRIVE_RUN}/weights/last.pt'
if os.path.exists(drive_last):
    os.makedirs(f'{LOCAL_RUN}/weights', exist_ok=True)
    for w in ['last.pt', 'best.pt']:
        s = f'{DRIVE_RUN}/weights/{w}'
        if os.path.exists(s): shutil.copy(s, f'{LOCAL_RUN}/weights/{w}')
    for extra in ['args.yaml', 'results.csv']:
        s = f'{DRIVE_RUN}/{extra}'
        if os.path.exists(s): shutil.copy(s, f'{LOCAL_RUN}/{extra}')
    resume = True
    print(f'🔄 ادامهٔ آموزش از چک‌پوینت درایو')
else:
    print('🆕 شروع آموزش تازه')

model = YOLO(f'{LOCAL_RUN}/weights/last.pt' if resume else MODEL)

# ---------- همگام‌سازی دوره‌ای وزن‌ها با درایو ----------
def sync_to_drive(trainer):
    try:
        e = int(getattr(trainer, 'epoch', 0)) + 1
        if e % SYNC_EVERY and e != getattr(trainer, 'epochs', 0):
            return
        os.makedirs(f'{DRIVE_RUN}/weights', exist_ok=True)
        for w in ['last.pt', 'best.pt']:
            s = f'{trainer.save_dir}/weights/{w}'
            if os.path.exists(s): shutil.copy(s, f'{DRIVE_RUN}/weights/{w}')
        for extra in ['results.csv', 'args.yaml']:
            s = f'{trainer.save_dir}/{extra}'
            if os.path.exists(s): shutil.copy(s, f'{DRIVE_RUN}/{extra}')
        print(f'   ☁️ epoch {e}: وزن‌ها روی درایو ذخیره شد')
    except Exception as ex:
        print(f'   ⚠️ همگام‌سازی درایو ناموفق: {ex}')

model.add_callback('on_fit_epoch_end', sync_to_drive)

# ---------- آموزش ----------
results = model.train(
    data      = YAML,
    epochs    = EPOCHS,
    imgsz     = IMGSZ,
    batch     = BATCH,
    device    = 0,
    workers   = 2,
    project   = RUNS_LOCAL,
    name      = RUN,
    exist_ok  = True,
    resume    = resume,

    # --- بهینه‌ساز و نرخ یادگیری ---
    optimizer = 'auto',
    lr0       = 0.001,      # پایین: fine-tune است نه آموزش از صفر
    lrf       = 0.01,
    warmup_epochs = 3,
    cos_lr    = True,
    patience  = 40,

    # --- augmentation ---
    hsv_h     = 0.015,      # ★ کم — رنگ آتش معنادار است
    hsv_s     = 0.7,
    hsv_v     = 0.5,        # ★ زیاد — شب و نور کم
    degrees   = 5.0,
    translate = 0.1,
    scale     = 0.5,
    shear     = 2.0,
    perspective = 0.0,
    flipud    = 0.0,        # آتش هرگز وارونه نیست
    fliplr    = 0.5,
    mosaic    = 1.0,
    close_mosaic = 15,      # ★ ۱۵ epoch آخر بدون mosaic
    mixup     = 0.10,
    erasing   = 0.20,

    # --- حافظه و ذخیره ---
    cache     = False,      # ۱۱۳ هزار تصویر در RAM جا نمی‌شود
    save_period = SYNC_EVERY,
    plots     = True,
    val       = True,
    amp       = True,

    # --- ★ قلّاب رسمی برای augmentation مخصوص CCTV ---
    **({'augmentations': CCTV_AUGS} if CCTV_AUGS else {}),
)

# آخرین همگام‌سازی
os.makedirs(f'{DRIVE_RUN}/weights', exist_ok=True)
for f in glob.glob(f'{LOCAL_RUN}/**/*', recursive=True):
    if os.path.isfile(f):
        rel = os.path.relpath(f, LOCAL_RUN)
        d = os.path.join(DRIVE_RUN, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy(f, d)
print(f'\n✅ آموزش تمام شد. همه‌چیز در درایو: {DRIVE_RUN}')
"""))

# ============================================================ 6 · ارزیابی
cells.append(md(r"""
## ۶ · ارزیابی

**نکته‌ای که باید بدانی:** mAP دود همیشه از mAP آتش پایین‌تر می‌آید.
این عیب مدل تو نیست — مرزِ دود ذاتاً مبهم است و حتی دو انسان متخصص هم برای یک ستون دود
کادر یکسان نمی‌کشند. پس همیشه **جدا** گزارششان کن.
"""))

cells.append(code(r"""
from ultralytics import YOLO
import pandas as pd, matplotlib.pyplot as plt, os

best = f'{LOCAL_RUN}/weights/best.pt'
if not os.path.exists(best): best = f'{DRIVE_RUN}/weights/best.pt'
model = YOLO(best)

m = model.val(data=YAML, split='test', imgsz=IMGSZ, conf=0.001, iou=0.6, plots=True)

print('\n' + '='*46)
print(f"{'معیار':22}{'مقدار':>12}")
print('-'*46)
print(f"{'mAP@50 (کل)':22}{m.box.map50:>12.4f}")
print(f"{'mAP@50-95 (کل)':22}{m.box.map:>12.4f}")
print(f"{'precision (کل)':22}{m.box.mp:>12.4f}")
print(f"{'recall (کل)':22}{m.box.mr:>12.4f}")
print('-'*46)
for i, nm in model.names.items():
    print(f"{'mAP@50 — '+nm:22}{m.box.ap50[i]:>12.4f}")
print('='*46)

# منحنی آموزش
csv = f'{LOCAL_RUN}/results.csv'
if os.path.exists(csv):
    df = pd.read_csv(csv); df.columns = df.columns.str.strip()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    for c in [c for c in df.columns if 'train/' in c and 'loss' in c]:
        ax[0].plot(df['epoch'], df[c], label=c.split('/')[-1])
    ax[0].set_title('خطای آموزش'); ax[0].legend(); ax[0].grid(alpha=.3)
    for c in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
        if c in df: ax[1].plot(df['epoch'], df[c], label=c.split('/')[-1])
    ax[1].set_title('mAP'); ax[1].legend(); ax[1].grid(alpha=.3)
    for c in ['metrics/precision(B)', 'metrics/recall(B)']:
        if c in df: ax[2].plot(df['epoch'], df[c], label=c.split('/')[-1])
    ax[2].set_title('precision / recall'); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.show()
"""))

# ============================================================ 7 · نقطهٔ کار
cells.append(md(r"""
## ۷ · ★ انتخاب نقطهٔ کار بر اساس **آلارم کاذب**

این بخشی است که یک مدل آکادمیک را به یک محصول تبدیل می‌کند.

`best.pt` را Ultralytics بر اساس **fitness** (ترکیب وزن‌دار mAP) انتخاب می‌کند.
ولی مشتری mAP نمی‌خرد — این را می‌پرسد:

> «شبی چند بار الکی زنگ می‌زند؟»

پس آستانهٔ اطمینان را روی **تصاویر منفی** تنظیم می‌کنیم: بالاترین recall را پیدا می‌کنیم
که نرخ آلارم کاذب هنوز زیر سقف قابل قبول باشد.
"""))

cells.append(code(r"""
import glob, os, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO(best)

# تصاویر منفی و مثبت مجموعهٔ آزمون را جدا کن
neg, pos = [], []
for ip in glob.glob(f'{DATA_LOCAL}/images/test/*'):
    lp = f"{DATA_LOCAL}/labels/test/{os.path.splitext(os.path.basename(ip))[0]}.txt"
    lines = [l for l in open(lp).read().splitlines() if l.strip()] if os.path.exists(lp) else []
    (pos if lines else neg).append(ip)
print(f'آزمون: {len(pos):,} تصویر دارای آتش/دود  ·  {len(neg):,} تصویر منفی')

SAMPLE = 1500          # برای سرعت؛ بالا ببر اگر وقت داری
neg_s, pos_s = neg[:SAMPLE], pos[:SAMPLE]

def max_conf(paths):
    # بیشترین اطمینان هر تصویر (صفر اگر هیچ چیزی ندید)
    out = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i+64], imgsz=IMGSZ, conf=0.01,
                               verbose=False, stream=False):
            c = r.boxes.conf
            out.append(float(c.max()) if c is not None and len(c) else 0.0)
    return np.array(out)

cn, cp = max_conf(neg_s), max_conf(pos_s)

print(f"\n{'آستانه':>8}{'نرخ آلارم کاذب':>16}{'نرخ تشخیص':>13}")
print('-'*40)
rows = []
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    far = (cn >= th).mean()
    tpr = (cp >= th).mean()
    rows.append((th, far, tpr))
    print(f'{th:>8.2f}{far*100:>15.2f}%{tpr*100:>12.1f}%')

# سقف: حداکثر ۱٪ فریم‌های منفی آلارم بدهند
TARGET_FAR = 0.01
ok = [r for r in rows if r[1] <= TARGET_FAR]
if ok:
    th, far, tpr = min(ok, key=lambda r: r[0])      # پایین‌ترین آستانه‌ای که سقف را رد نمی‌کند
    print(f'\n★ آستانهٔ پیشنهادی = {th:.2f}   (آلارم کاذب {far*100:.2f}% · تشخیص {tpr*100:.1f}%)')
else:
    th = 0.5
    print(f'\n⚠️ هیچ آستانه‌ای به آلارم کاذب ≤{TARGET_FAR*100:.0f}% نرسید — {th} را موقتاً بگذار و داده‌ی منفی بیشتری لازم است')
RECOMMENDED_CONF = th

plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.hist(cn, bins=50, alpha=.7, label='منفی (نباید زنگ بزند)')
plt.hist(cp, bins=50, alpha=.7, label='مثبت (باید زنگ بزند)')
plt.axvline(th, color='r', ls='--', label=f'آستانه {th:.2f}')
plt.legend(); plt.title('توزیع بیشترین اطمینان'); plt.yscale('log')
plt.subplot(1, 2, 2)
plt.plot([r[1]*100 for r in rows], [r[2]*100 for r in rows], 'o-')
plt.xlabel('آلارم کاذب %'); plt.ylabel('تشخیص %'); plt.grid(alpha=.3)
plt.title('منحنی کار')
plt.tight_layout(); plt.show()
"""))

# ============================================================ 8 · سرعت
cells.append(md(r"""
## ۸ · خروجی صنعتی و **سنجش واقعی سرعت**

### جواب صادقانه به «آیا سریع‌تر هم می‌شویم؟»

باید دو چیز را از هم جدا کرد:

| منبع سرعت | آیا کمک می‌کند؟ |
|---|---|
| **خودِ fine-tune** | ❌ **هیچ.** معماری و FLOPs عوض نمی‌شود. وزن‌ها عوض می‌شوند، نه سرعت. |
| سر مدل با ۲ کلاس به‌جای ۸۰ | ✅ کمی — سرِ تشخیص کوچک‌تر می‌شود (اثر کم ولی واقعی) |
| YOLO26 بدون NMS | ✅ NMS روی CPU اجرا می‌شود و batch نمی‌شود؛ حذفش تأخیر را **قابل پیش‌بینی** می‌کند |
| مدل کوچک‌تر (n به‌جای s) | ✅ زیاد — با هزینهٔ دقت |
| `imgsz` پایین‌تر | ✅ زیاد — FLOPs با مربع اندازه رشد می‌کند |
| **TensorRT FP16 / INT8** | ✅✅ **بیشترین برد** — چند برابر |

پس: **بله سریع‌تر می‌شویم، ولی از مسیر خروجی‌گرفتن و استقرار، نه از آموزش.**
سلول زیر همهٔ اینها را واقعاً اندازه می‌گیرد، نه حدس.
"""))

cells.append(code(r"""
import time, torch, os
from ultralytics import YOLO

def bench(m, imgsz, n=40, half=False):
    x = torch.zeros(1, 3, imgsz, imgsz)
    for _ in range(8): m.predict(x, imgsz=imgsz, half=half, verbose=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t = time.time()
    for _ in range(n): m.predict(x, imgsz=imgsz, half=half, verbose=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.time()-t)/n*1000

rows = []
tuned = YOLO(best)
rows.append(('YOLO26 خام (COCO, ۸۰ کلاس)', f'{IMGSZ} PyTorch', BASE_MS))
rows.append(('مدل ما (۲ کلاس)',            f'{IMGSZ} PyTorch', bench(tuned, IMGSZ)))
rows.append(('مدل ما — FP16',              f'{IMGSZ} PyTorch', bench(tuned, IMGSZ, half=True)))
rows.append(('مدل ما — imgsz 512',         '512 PyTorch',      bench(tuned, 512)))
rows.append(('مدل ما — imgsz 416',         '416 PyTorch',      bench(tuned, 416)))

# ---------- ONNX ----------
try:
    p = tuned.export(format='onnx', imgsz=IMGSZ, simplify=True, dynamic=False)
    rows.append(('مدل ما — ONNX', f'{IMGSZ} onnxruntime', bench(YOLO(p), IMGSZ)))
except Exception as e:
    print('⚠️ خروجی ONNX نگرفت:', e)

# ---------- TensorRT ----------
try:
    p = tuned.export(format='engine', imgsz=IMGSZ, half=True)     # چند دقیقه طول می‌کشد
    rows.append(('مدل ما — TensorRT FP16', f'{IMGSZ} TensorRT', bench(YOLO(p), IMGSZ)))
except Exception as e:
    print('⚠️ TensorRT نگرفت (روی Colab گاهی در دسترس نیست):', e)

print('\n' + '='*74)
print(f"{'حالت':34}{'اجرا':22}{'ms/frame':>9}{'FPS':>9}")
print('-'*74)
for nm, mode, ms in rows:
    print(f'{nm:34}{mode:22}{ms:>9.1f}{1000/ms:>9.1f}')
print('='*74)
print(f'\nمرجع خام: {BASE_MS:.1f} ms')
print('یادآوری: fine-tune سرعت را عوض نمی‌کند — خروجی‌گرفتن و اندازهٔ ورودی عوض می‌کنند.')
"""))

cells.append(code(r"""
# ---------- ذخیرهٔ همه‌چیز در درایو ----------
import shutil, os, glob, json

os.makedirs(f'{DRIVE_RUN}/export', exist_ok=True)
for pat in ['*.onnx', '*.engine', '*.torchscript']:
    for f in glob.glob(f'{LOCAL_RUN}/weights/{pat}'):
        shutil.copy(f, f'{DRIVE_RUN}/export/{os.path.basename(f)}')

card = {
    'run': RUN,
    'base_model': MODEL,
    'imgsz': IMGSZ,
    'epochs': EPOCHS,
    'classes': {0: 'fire', 1: 'smoke'},
    'recommended_conf': float(RECOMMENDED_CONF),
    'mAP50': float(m.box.map50),
    'mAP50_95': float(m.box.map),
    'mAP50_fire': float(m.box.ap50[0]),
    'mAP50_smoke': float(m.box.ap50[1]),
    'latency_ms': {nm: round(ms, 2) for nm, _, ms in rows},
    'notes': 'CCTV augmentation فعال بود: فشرده‌سازی، رزولوشن پایین، خاکستری، بلور',
}
with open(f'{DRIVE_RUN}/model_card.json', 'w') as f:
    json.dump(card, f, indent=2, ensure_ascii=False)

print('✅ ذخیره شد در درایو:')
print(f'   وزن‌ها   : {DRIVE_RUN}/weights/best.pt')
print(f'   خروجی‌ها : {DRIVE_RUN}/export/')
print(f'   شناسنامه: {DRIVE_RUN}/model_card.json')
print()
print(json.dumps(card, indent=2, ensure_ascii=False))
"""))

# ============================================================ 9 · بعد
cells.append(md(r"""
## بعد از این

وزن نهایی: `FireGuard_Runs/<RUN>/weights/best.pt` — با `conf` پیشنهادیِ بخش ۷ استفاده کن.

**قدم بعدی که بیشترین ارزش را دارد:** ~۱۲ ساعت ویدیوی خام از دوربین واقعی خودت
(بدون برچسب) ضبط کن و بخش ۷ را روی آن اجرا کن. آن‌وقت برای اولین بار عددِ
«شبی چند بار الکی زنگ می‌زند» را داری — و همان عددی است که مشتری می‌خرد.

بعد از آن: BLAZE (ردیابی + تأییدکنندهٔ سوسوی DFT + رشد ستون دود) طبق `PLAN_V2.md`.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"ساخته شد: {NB}  ({len(cells)} سلول, {os.path.getsize(NB)/1024:.0f} KB)")
