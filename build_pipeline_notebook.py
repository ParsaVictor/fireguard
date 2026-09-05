# -*- coding: utf-8 -*-
"""سازندهٔ نوت‌بوک یکپارچهٔ FireGuard (داده + آموزش + ارزیابی + خروجی).
نوت‌بوک را دستی ویرایش نکن — این فایل را عوض کن و دوباره اجرا کن:
    py -3 build_pipeline_notebook.py
"""
import json
import os

NB = "FireGuard_Pipeline.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


C = []

# ══════════════════════════════════════════════════════ ۰ · نقشه
C.append(md(r"""
# 🔥 FireGuard — خط لولهٔ کامل

از دیتاست خام تا مدلِ آمادهٔ استقرار، **در یک نوت‌بوک**.
دامنه: **دوربین مداربستهٔ داخلی** (خانه · دفتر · مغازه). دو کلاس: `fire` و `smoke`.

---

## نقشهٔ بخش‌ها

| # | بخش | هر جلسه؟ | زمان |
|---|---|---|---|
| ۱ | نصب + شناساییِ خودکارِ سخت‌افزار | ✅ | ۲ دقیقه |
| ۲ | **تنظیمات** ← تنها سلولی که دست می‌زنی | ✅ | — |
| ۳ | دانلودِ **موازی** دیتاست‌ها | بار اول | ۵-۱۵ دقیقه |
| ۴ | ادغام · اصلاح کلاس · حذف تکراری · تقسیم | بار اول | ۱۰ دقیقه |
| ۵ | **بازرسی داده** | بار اول | ۱ دقیقه |
| ۶ | 🚁 **پرواز آزمایشی** — کل خط لوله در ۳ دقیقه | بار اول | ۳ دقیقه |
| ۷ | خط پایه — مدل خام | بار اول | ۱ دقیقه |
| ۸ | 🔬 **آزمون A/B روی augmentation** | اختیاری | ۴۰ دقیقه |
| ۹ | **آموزش اصلی** (رزولوشن پلکانی + resume) | ✅ | ۵-۸ ساعت |
| ۱۰ | ارزیابی | بعد | ۵ دقیقه |
| ۱۱ | **انتخاب نقطهٔ کار بر اساس آلارم کاذب** | بعد | ۵ دقیقه |
| ۱۲ | خروجی صنعتی + سنجش واقعی سرعت | بعد | ۱۰ دقیقه |
| ۱۳ | ذخیره در درایو + شناسنامهٔ مدل | بعد | ۱ دقیقه |

---

## دربارهٔ «فقط ۲ کلاس»

مدلِ پایه (`yolo26s.pt`) روی COCO با ۸۰ کلاس آموزش دیده. وقتی روی `fire.yaml`
با `nc: 2` آموزشش می‌دهیم، Ultralytics **سرِ ۸۰کلاسه را دور می‌ریزد و سرِ ۲کلاسه
می‌سازد** — در لاگ می‌بینی:

```
Overriding model.yaml nc=80 with nc=2
```

آن ۸۰ کلاس فقط برای وزنِ اولیهٔ backbone به کار می‌رود (لبه، بافت، گرادیان رنگ —
که برای گربه و شعله یکی است). **مدلِ نهایی فقط `fire` و `smoke` دارد.**

---

## ⚠️ تله‌هایی که این نوت‌بوک از قبل حلشان کرده

| # | تله | راه‌حل |
|---|---|---|
| ۱ | **نگاشت کلاس برعکس** — FASDD `0=fire` ولی D-Fire `0=smoke` | برچسب‌های D-Fire با `1-c` جابه‌جا می‌شوند |
| ۲ | **فریم‌های متوالی** — بخش AoF فریم‌های یک ویدیوست (۸۰-۹۰٪ شبیه) | تقسیم گروه‌آگاه |
| ۳ | **فریم‌های هم‌رویداد از تور رد می‌شدند** — فاصلهٔ ۶-۱۳ بیت، دورتر از آستانهٔ حذف (۳) | آستانهٔ دوم: ۴..۱۲ = «هم‌رویداد» → یک split |
| ۴ | **سطل‌های غول‌پیکر هش** — ۳۹ هزار منفیِ ساده هشِ یکسان می‌گیرند و از حذف تکراری جا می‌ماندند | گام هشِ دقیق در O(n) قبل از جست‌وجوی نزدیک |
| ۵ | **`imgsz>640` بی‌فایده** — تصاویر FASDD خودشان حداکثر ۶۴۰ پیکسل‌اند | سقفِ ۶۴۰ |
| ۶ | **`resume` همهٔ آرگومان‌ها را از چک‌پوینت بازمی‌نویسد** (`self.args = get_cfg(ckpt_args)`) | هر مرحلهٔ رزولوشن یک `train()` تازه است، نه resume |
| ۷ | **درایو با فایل کوچکِ زیاد کند است** | داده و وزن روی `/content`؛ همگام‌سازی در نخِ پس‌زمینه |
"""))

# ══════════════════════════════════════════════════════ ۱ · نصب
C.append(md(r"""
## ۱ · نصب و شناساییِ خودکارِ سخت‌افزار

Colab هر بار ماشینِ متفاوتی می‌دهد (۲ تا ۸ هسته، T4 یا L4 یا A100).
به‌جای عددِ ثابت، `workers` و `batch` از روی همان ماشین حساب می‌شوند.
"""))

C.append(code(r"""
# ═══ بستهٔ اصلی ═══
!pip install -q --upgrade ultralytics

# ═══ augmentation ═══
!pip install -q albumentations

# ═══ ارزیابی و رسم ═══
!pip install -q "matplotlib>=3.8" pandas seaborn

# ═══ خروجی صنعتی ═══
!pip install -q onnx onnxslim onnxruntime-gpu

import os, torch, ultralytics
print('ultralytics :', ultralytics.__version__)
print('torch       :', torch.__version__, '| CUDA:', torch.cuda.is_available())

# ---------- شناسایی سخت‌افزار ----------
N_CPU = os.cpu_count() or 2
HAS_GPU = torch.cuda.is_available()
GPU_NAME, GPU_GB = 'CPU', 0
if HAS_GPU:
    p = torch.cuda.get_device_properties(0)
    GPU_NAME, GPU_GB = p.name, p.total_memory / 1e9

# workers: پیش‌فرض Ultralytics ۸ است؛ روی Colab دوهسته‌ای باعث گیرکردن می‌شود
AUTO_WORKERS = max(2, min(8, N_CPU - 1))

print(f'\nCPU  : {N_CPU} هسته        → workers = {AUTO_WORKERS}')
print(f'GPU  : {GPU_NAME}  ({GPU_GB:.1f} GB)')
if not HAS_GPU:
    print('\n⚠️  GPU روشن نیست:  Runtime → Change runtime type → T4 GPU')
!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null | head -2
!df -h /content | tail -1
"""))

# ══════════════════════════════════════════════════════ ۲ · تنظیمات
C.append(md(r"""
## ۲ · تنظیمات

**تنها سلولی که دست می‌زنی.** بقیه را فقط اجرا کن.
"""))

C.append(code(r"""
from google.colab import drive
drive.mount('/content/drive')
import os

# ═════════════════ مسیرها ═════════════════
DRIVE_DATA = '/content/drive/MyDrive/FireGuard_Datasets'   # آرشیو zip دیتاست‌ها
DRIVE_RUNS = '/content/drive/MyDrive/FireGuard_Runs'       # ★ وزن‌ها اینجا ذخیره می‌شوند

# ═════════════════ مدل ═════════════════
MODEL  = 'yolo26s.pt'      # yolo26n سریع‌ترین · yolo26s تعادل (پیشنهاد) · yolo26m دقیق‌تر
RUN    = 'fireguard_s_v1'  # برای آزمایش جدید عوضش کن

# ═════════════════ آموزش ═════════════════
EPOCHS      = 120
PROGRESSIVE = True    # ★ مرحلهٔ ۱ در ۵۱۲ (سریع) سپس مرحلهٔ ۲ در ۶۴۰ (دقیق)
IMGSZ_HI    = 640     # بالاتر نبر — تصاویر FASDD حداکثر ۶۴۰ پیکسل‌اند
IMGSZ_LO    = 512
STAGE1_FRAC = 0.60    # چند درصد epoch ها در رزولوشن پایین

BATCH   = -1          # -1 = AutoBatch (خودش بیشترین اندازهٔ ممکن را پیدا می‌کند)
WORKERS = None        # None = خودکار از روی تعداد هسته

# ═════════════════ گزینه‌ها ═════════════════
CCTV_AUG    = True    # augmentation مخصوص دوربین مداربسته
RUN_SMOKETEST = True  # 🚁 پرواز آزمایشی قبل از آموزش اصلی — خاموشش نکن
RUN_ABLATION  = False # 🔬 آزمون A/B روی augmentation (۴۰ دقیقه) — بخش ۸

# ═════════════════ داده ═════════════════
USE_FASDD, USE_DFIRE = True, True
DEDUP, DEDUP_DISTANCE, GROUP_DISTANCE = True, 3, 12
SYNC_EVERY = 5        # هر چند epoch وزن به درایو

# ══════════════════════════════════════════
WORK      = '/content/work'
DATA_DIR  = '/content/fireguard_data'
RUNS_LOCAL= '/content/runs'
DRIVE_RUN = f'{DRIVE_RUNS}/{RUN}'
YAML      = f'{DATA_DIR}/fire.yaml'
if WORKERS is None: WORKERS = AUTO_WORKERS

for d in (DRIVE_DATA, DRIVE_RUNS, DRIVE_RUN, WORK, RUNS_LOCAL):
    os.makedirs(d, exist_ok=True)

print(f'مدل        : {MODEL}')
print(f'اجرا       : {RUN}')
print(f'epochs     : {EPOCHS}' + (f'  (پلکانی: {int(EPOCHS*STAGE1_FRAC)} @{IMGSZ_LO} + {EPOCHS-int(EPOCHS*STAGE1_FRAC)} @{IMGSZ_HI})' if PROGRESSIVE else f'  @{IMGSZ_HI}'))
print(f'batch      : {"AutoBatch" if BATCH==-1 else BATCH}   workers: {WORKERS}')
print(f'وزن‌ها     → {DRIVE_RUN}')
"""))

# ══════════════════════════════════════════════════════ ۳ · دانلود موازی
C.append(md(r"""
## ۳ · دانلودِ **موازی** دیتاست‌ها

FASDD (۳.۴ گیگ، یک فایل) و D-Fire (۲۱ هزار فایل کوچک) **همزمان** دانلود می‌شوند.
چون گلوگاهِ یکی پهنای‌باند و دیگری تعداد درخواست است، موازی‌کردن واقعاً وقت می‌برد
— نه اینکه فقط شلوغ به‌نظر برسد.

zip ها در درایو می‌مانند و **باز نمی‌شوند**؛ درایو با ۹۵ هزار فایل کوچک
ساعت‌ها طول می‌کشد و اغلب وسط راه می‌شکند.
"""))

C.append(code(r"""
# ═══════════════════════════════════════════════════════════════════════
#  دانلود مقاوم — نسخهٔ اصلاح‌شده بعد از خطای 429
#
#  چه شد: snapshot_download داشت ۱۴,۶۱۴ فایلِ جدا با ۱۶ نخ می‌گرفت و برای
#  هر کدام یک درخواستِ توکن به HF می‌زد → HTTP 429 Too Many Requests.
#  (مشکل اینترنت نبود.)
#
#  راه‌حل: همان دیتاست نسخهٔ parquet هم دارد — **۱۲ فایل به‌جای ۱۴,۶۱۴**.
#  فیلد `label` دقیقاً همان متنِ برچسبِ YOLO است، پس چیزی از دست نمی‌رود.
# ═══════════════════════════════════════════════════════════════════════
import os, sys, time, shutil, glob

# xet همان مسیری است که 429 داد؛ خاموشش می‌کنیم و ماژول را دوباره بار می‌زنیم
os.environ['HF_HUB_DISABLE_XET'] = '1'
for _m in [k for k in list(sys.modules) if k.startswith('huggingface_hub')]:
    del sys.modules[_m]
from huggingface_hub import hf_hub_download

!pip install -q pyarrow

def human(n):
    for u in ['B','KB','MB','GB']:
        if n < 1024: return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}TB'

def retry(fn, tries=6, base=5, what=''):
    # تلاش دوباره با عقب‌نشینی نمایی — 429 با صبرکردن حل می‌شود
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1: raise
            w = base * (2 ** i)
            print(f'      ⏳ {what} تلاش {i+1}/{tries} ناموفق ({type(e).__name__}) — {w}s صبر')
            time.sleep(w)

# ───────────────────────── FASDD: یک فایل، بدون دردسر ─────────────────────────
def get_fasdd():
    dst = f'{DRIVE_DATA}/FASDD_CV.zip'
    if os.path.exists(dst) and os.path.getsize(dst) > 3.3e9:
        return f'✅ FASDD از قبل هست ({human(os.path.getsize(dst))})'
    t = time.time()
    p = retry(lambda: hf_hub_download('seawsurf/fire_smoke_dataset_fasdd_cv', 'FASDD_CV.zip',
                                      repo_type='dataset', local_dir='/content/dl'),
              what='FASDD')
    shutil.move(p, dst)
    return f'✅ FASDD ({human(os.path.getsize(dst))}) در {time.time()-t:.0f}s'

# ───────────────── D-Fire: ۱۲ فایل parquet، پشتِ‌سرِهم ─────────────────
DF_PARQUET = f'{DRIVE_DATA}/D-Fire_parquet'
DF_OUT     = f'{WORK}/dfire'

def get_dfire():
    if os.path.isdir(f'{DF_OUT}/train/images') and os.listdir(f'{DF_OUT}/train/images'):
        return '✅ D-Fire از قبل استخراج شده'
    os.makedirs(DF_PARQUET, exist_ok=True)
    files = ([f'data/train-{i:05d}-of-00009.parquet' for i in range(9)]
             + [f'data/test-{i:05d}-of-00003.parquet' for i in range(3)])

    t = time.time(); got = 0
    for i, rel in enumerate(files, 1):
        dst = f'{DF_PARQUET}/{os.path.basename(rel)}'
        if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
            continue                                  # ★ از قبل در درایو — دوباره دانلود نمی‌شود
        print(f'   [{i}/12] {os.path.basename(rel)} ...')
        p = retry(lambda: hf_hub_download('badsaarow/d-fire', rel, repo_type='dataset',
                                          local_dir='/content/dl/dfp'),
                  what=f'D-Fire {i}/12')
        shutil.move(p, dst); got += 1
        time.sleep(1)                                 # نفس‌کشیدن بین درخواست‌ها
    msg_dl = f'{got} فایل تازه' if got else 'همه از قبل در درایو'
    print(f'   parquet آماده ({msg_dl}) در {time.time()-t:.0f}s — استخراج ...')

    # ---- parquet → images/ + labels/ (همان ساختاری که سلول ادغام انتظار دارد) ----
    import pyarrow.parquet as pq
    for sp in ['train','test']:
        os.makedirs(f'{DF_OUT}/{sp}/images', exist_ok=True)
        os.makedirs(f'{DF_OUT}/{sp}/labels', exist_ok=True)
    n = 0
    for pf_path in sorted(glob.glob(f'{DF_PARQUET}/*.parquet')):
        sp = 'train' if 'train-' in os.path.basename(pf_path) else 'test'
        for batch in pq.ParquetFile(pf_path).iter_batches(batch_size=128):
            d = batch.to_pydict()
            for img, lab, fn in zip(d['image'], d['label'], d['filename']):
                raw = img['bytes'] if isinstance(img, dict) else img
                if not raw: continue
                fn = os.path.basename(fn); stem = os.path.splitext(fn)[0]
                open(f'{DF_OUT}/{sp}/images/{fn}', 'wb').write(raw)
                open(f'{DF_OUT}/{sp}/labels/{stem}.txt', 'w').write(lab or '')
                n += 1
    return f'✅ D-Fire — {n:,} تصویر استخراج شد در {time.time()-t:.0f}s'

# ───────────────────────── اجرا: دو دیتاست موازی ─────────────────────────
from concurrent.futures import ThreadPoolExecutor
jobs = ([get_fasdd] if USE_FASDD else []) + ([get_dfire] if USE_DFIRE else [])
print(f'⬇️  {len(jobs)} دیتاست موازی (حداکثر ۲ اتصال همزمان به HF)')
print()
t0 = time.time()
with ThreadPoolExecutor(len(jobs) or 1) as ex:
    for r in ex.map(lambda f: f(), jobs):
        print('  ', r)
print()
print(f'مجموع: {time.time()-t0:.0f} ثانیه')

try:
    fr = shutil.disk_usage('/content/drive/MyDrive').free
    print()
    print(f'فضای آزاد درایو: {human(fr)}')
    if fr < 6e9:
        print('⚠️  کمتر از ۶ گیگ آزاد است. بعد از ساخته‌شدن fireguard_data.zip می‌توانی')
        print('    پوشهٔ D-Fire_parquet را پاک کنی (۳.۱ گیگ) — دیگر لازمش نداری.')
except Exception:
    pass
!ls -lh "$DRIVE_DATA" 2>/dev/null
"""))

C.append(md(r"""
### باز کردن روی دیسک محلی — **این را هر جلسه اجرا کن**

`/content` با قطع‌شدن Colab پاک می‌شود. این سلول از درایو کپی می‌کند، نه دانلود دوباره.
"""))

C.append(code(r"""
import zipfile, os, time
from concurrent.futures import ThreadPoolExecutor

def unzip(args):
    name, sub = args
    src, out = f'{DRIVE_DATA}/{name}', f'{WORK}/{sub}'
    if os.path.isdir(out) and os.listdir(out):
        return f'✅ {sub} از قبل باز شده', out
    if not os.path.exists(src):
        return f'⏭️  {name} نیست', None
    t = time.time()
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(src) as z: z.extractall(out)
    return f'✅ {sub} در {time.time()-t:.0f}s', out

todo = ([('FASDD_CV.zip','fasdd')] if USE_FASDD else []) + \
       ([('D-Fire.zip','dfire')] if USE_DFIRE else [])
res = {}
with ThreadPoolExecutor(len(todo) or 1) as ex:
    for (msg, out), (name, sub) in zip(ex.map(unzip, todo), todo):
        print('  ', msg); res[sub] = out

FASDD_DIR, DFIRE_DIR = res.get('fasdd'), res.get('dfire')
!df -h /content | tail -1
"""))

# ══════════════════════════════════════════════════════ ۴ · ادغام
C.append(md(r"""
## ۴ · ادغام · اصلاح کلاس · حذف تکراری · تقسیم

### ★ تلهٔ ۱ — نگاشت کلاس

| منبع | در فایل اصلی | تبدیل ما |
|---|---|---|
| FASDD | `0=fire`, `1=smoke` | بدون تغییر ✅ |
| D-Fire | `0=smoke`, `1=fire` | **۰ ↔ ۱ جابه‌جا** 🔄 |

خروجی همه‌جا: **`0 = fire` · `1 = smoke`**

منفی‌ها با فایل برچسبِ **خالی** نگه داشته می‌شوند — درست است، و همان‌ها آلارم کاذب را
پایین می‌آورند. دور نریزشان.
"""))

C.append(code(r"""
import os, glob, re, collections

records, IMG_EXT = [], ('.jpg','.jpeg','.png','.JPG','.PNG')

def read_label(p):
    if not p or not os.path.exists(p): return []
    out = []
    for line in open(p, encoding='utf-8', errors='replace').read().strip().splitlines():
        t = line.split()
        if len(t) >= 5:
            out.append((int(float(t[0])), *[float(x) for x in t[1:5]]))
    return out

def scan(root, splits, source, flip):
    n, nb = 0, 0
    for sp in splits:
        idir = os.path.join(root, sp, 'images')
        if not os.path.isdir(idir): continue
        for ip in glob.glob(os.path.join(idir, '*')):
            if not ip.endswith(IMG_EXT): continue
            stem = os.path.splitext(os.path.basename(ip))[0]
            lab = read_label(os.path.join(root, sp, 'labels', stem + '.txt'))
            if flip:
                lab = [(1 - c, *r) for (c, *r) in lab]; nb += len(lab)
            if source == 'fasdd':
                m = re.match(r'(.+?)_jpg\.rf\.', os.path.basename(ip))
                grp = m.group(1) if m else stem
            else:
                m = re.match(r'([A-Za-z]+)(\d+)', stem)
                grp = f'{m.group(1)}{m.group(2)[:-2]}' if m and len(m.group(2)) > 2 else stem
            records.append(dict(src=ip, lab=lab, source=source, group=grp))
            n += 1
    return n, nb

if FASDD_DIR:
    n, _ = scan(FASDD_DIR, ['train','valid','test'], 'fasdd', False)
    print(f'FASDD  : {n:,} تصویر   (نگاشت بدون تغییر)')
if DFIRE_DIR:
    n, nb = scan(DFIRE_DIR, ['train','test'], 'dfire', True)
    print(f'D-Fire : {n:,} تصویر   ({nb:,} کادر جابه‌جا شد ۰↔۱) 🔄')

c = collections.Counter()
for r in records:
    cs = {x[0] for x in r['lab']}
    if 0 in cs: c['fire'] += 1
    if 1 in cs: c['smoke'] += 1
    if not r['lab']: c['neg'] += 1
print(f"\nمجموع {len(records):,} تصویر  |  آتش‌دار {c['fire']:,}  دوددار {c['smoke']:,}  منفی {c['neg']:,}")
"""))

C.append(md(r"""
### ★ تله‌های ۳ و ۴ — حذف تکراری، دو مرحله‌ای و دو آستانه‌ای

| فاصلهٔ همینگ dHash | معنی | رفتار |
|---|---|---|
| ≤ ۳ | تکراری | یکی حذف می‌شود |
| **۴ تا ۱۲** | **هم‌رویداد** | هر دو می‌مانند، **در یک split قفل می‌شوند** |
| > ۱۲ | مستقل | آزاد |

آستانهٔ دوم لازم است چون فریم‌های متوالیِ اندازه‌گیری‌شده ۸۰-۹۰٪ شبیه بودند
= **۶ تا ۱۳ بیت** اختلاف — خیلی دورتر از آستانهٔ حذف.

و گامِ «هشِ دقیق» اول لازم است چون ۳۹ هزار منفیِ ساده (آسمان، دیوار) هشِ یکسان
می‌گیرند و یک سطلِ غول‌پیکر می‌سازند که جست‌وجوی جفتی را از کار می‌اندازد.
"""))

C.append(code(r"""
import numpy as np, collections
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def dhash64(p):
    try:
        a = np.asarray(Image.open(p).convert('L').resize((9,8)), dtype=np.int16)
        v = 0
        for b in (a[:,1:] > a[:,:-1]).flatten(): v = (v<<1) | int(b)
        return v
    except Exception:
        return None

KEY_PARENT = {}
def kfind(k):
    KEY_PARENT.setdefault(k, k)
    while KEY_PARENT[k] != k:
        KEY_PARENT[k] = KEY_PARENT[KEY_PARENT[k]]; k = KEY_PARENT[k]
    return k
def kunion(a, b):
    ra, rb = kfind(a), kfind(b)
    if ra != rb: KEY_PARENT[rb] = ra
def gkey(r): return f"{r['source']}:{r['group']}"

if DEDUP:
    print(f'محاسبهٔ hash برای {len(records):,} تصویر (موازی، {WORKERS*4} نخ) ...')
    with ThreadPoolExecutor(WORKERS*4) as ex:
        for r, h in zip(records, ex.map(lambda r: dhash64(r['src']), records)): r['h'] = h
    ok = [r for r in records if r['h'] is not None]
    if len(ok) != len(records): print(f'   ⚠️ {len(records)-len(ok)} تصویر خوانده نشد')

    par = list(range(len(ok)))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: par[max(ra,rb)] = min(ra,rb)

    # گام ۱ — هشِ دقیقاً یکسان، O(n)
    exact = collections.defaultdict(list)
    for i, r in enumerate(ok): exact[r['h']].append(i)
    cross_exact = 0
    for _, ix in exact.items():
        if len({ok[j]['source'] for j in ix}) > 1: cross_exact += len(ix)-1
        for j in ix[1:]: union(ix[0], j)
    uniq = [ix[0] for ix in exact.values()]
    print(f'   هشِ یکتا {len(uniq):,} از {len(ok):,}  (کاملاً یکسان: {len(ok)-len(uniq):,})')

    # گام ۲ — نزدیک‌به‌هم، فقط روی هش‌های یکتا
    buckets = [collections.defaultdict(list) for _ in range(4)]
    for i in uniq:
        for k in range(4): buckets[k][(ok[i]['h'] >> (16*k)) & 0xFFFF].append(i)
    NEAR, cross, skipped, seen = [], 0, 0, set()
    for k in range(4):
        for _, ix in buckets[k].items():
            if len(ix) < 2: continue
            if len(ix) > 6000: skipped += len(ix); continue
            for a_i in range(len(ix)):
                for b_i in range(a_i+1, len(ix)):
                    a, b = ix[a_i], ix[b_i]
                    pk = (a,b) if a < b else (b,a)
                    if pk in seen: continue
                    seen.add(pk)
                    d = bin(ok[a]['h'] ^ ok[b]['h']).count('1')
                    if d <= DEDUP_DISTANCE:
                        if ok[a]['source'] != ok[b]['source']: cross += 1
                        union(a, b)
                    elif d <= GROUP_DISTANCE:
                        NEAR.append((a,b))
    if skipped: print(f'   ⚠️ {skipped:,} تصویر در سطل بزرگ جا ماند')

    clusters = collections.defaultdict(list)
    for i in range(len(ok)): clusters[find(i)].append(i)
    kept = []
    for _, mem in clusters.items():
        best = max(mem, key=lambda i: (len(ok[i]['lab']), ok[i]['source']=='fasdd'))
        kept.append(ok[best])
        for i in mem: kunion(gkey(ok[mem[0]]), gkey(ok[i]))
    for a, b in NEAR: kunion(gkey(ok[a]), gkey(ok[b]))

    print(f"   {len(NEAR):,} جفت «هم‌رویداد» → در یک split قفل شدند")
    print(f'✅ {len(ok)-len(kept):,} تکراری حذف شد ({100*(len(ok)-len(kept))/max(len(ok),1):.1f}%)'
          f'  ·  {cross+cross_exact:,} مورد بین دو دیتاست')
    records = kept
print(f'باقی‌مانده: {len(records):,}')
"""))

C.append(md(r"""
### ★ تلهٔ ۲ — تقسیم گروه‌آگاه

همهٔ فریم‌های یک رویداد اجباراً در **یک** split. وگرنه فریم ۱۲ در train و ۱۳ در test
می‌افتد و mAP دروغین بالا می‌آید.
"""))

C.append(code(r"""
import os, shutil, random, collections
from concurrent.futures import ThreadPoolExecutor

random.seed(1337)
shutil.rmtree(DATA_DIR, ignore_errors=True)
for s in ['train','val','test']:
    os.makedirs(f'{DATA_DIR}/images/{s}', exist_ok=True)
    os.makedirs(f'{DATA_DIR}/labels/{s}', exist_ok=True)

keyed = collections.defaultdict(list)
for r in records: keyed[kfind(gkey(r))].append(r)
keys = sorted(keyed); random.shuffle(keys)
n = len(keys); b1, b2 = int(0.80*n), int(0.90*n)
for i, k in enumerate(keys):
    sp = 'train' if i < b1 else ('val' if i < b2 else 'test')
    for r in keyed[k]: r['final'] = sp
print(f'تقسیم گروه‌آگاه روی {n:,} گروه')

def write(a):
    i, r = a
    ext = os.path.splitext(r['src'])[1].lower()
    nm = f"{r['source']}_{i:07d}"
    shutil.copy(r['src'], f"{DATA_DIR}/images/{r['final']}/{nm}{ext}")
    with open(f"{DATA_DIR}/labels/{r['final']}/{nm}.txt", 'w') as f:
        for c,x,y,w,h in r['lab']: f.write(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

print(f'نوشتن فایل‌ها (موازی، {WORKERS*4} نخ) ...')
with ThreadPoolExecutor(WORKERS*4) as ex:
    list(ex.map(write, enumerate(records)))

with open(YAML, 'w') as f:
    f.write('\n'.join([f'path: {DATA_DIR}', 'train: images/train', 'val: images/val',
                       'test: images/test', '', 'nc: 2', 'names:', '  0: fire', '  1: smoke', '']))

# صحت‌سنجی: هیچ گروهی نباید بین دو split پخش شده باشد
byk = collections.defaultdict(set)
for r in records: byk[kfind(gkey(r))].add(r['final'])
strad = sum(1 for v in byk.values() if len(v) > 1)
print(f'✅ گروه‌های پخش‌شده بین چند split: {strad}   (باید صفر باشد)')

st = collections.defaultdict(collections.Counter)
for r in records:
    d = st[r['final']]; cs = {x[0] for x in r['lab']}
    d['img'] += 1
    if 0 in cs: d['if'] += 1
    if 1 in cs: d['is'] += 1
    if not r['lab']: d['neg'] += 1
    for x in r['lab']: d['bf' if x[0]==0 else 'bs'] += 1
print('\n' + '='*62)
print(f"{'split':7}{'تصویر':>9}{'آتش‌دار':>10}{'دوددار':>10}{'منفی':>9}{'کادر آتش':>11}{'کادر دود':>11}")
print('-'*62)
for s in ['train','val','test']:
    d = st[s]
    print(f"{s:7}{d['img']:>9,}{d['if']:>10,}{d['is']:>10,}{d['neg']:>9,}{d['bf']:>11,}{d['bs']:>11,}")
print('='*62)

# آرشیو برای جلسه‌های بعد
print('\n📦 بسته‌بندی برای درایو ...')
shutil.make_archive('/content/fireguard_data', 'zip', DATA_DIR)
shutil.move('/content/fireguard_data.zip', f'{DRIVE_DATA}/fireguard_data.zip')
print('✅', f'{DRIVE_DATA}/fireguard_data.zip')
"""))

# ══════════════════════════════════════════════════════ ۵ · بازرسی
C.append(md(r"""
## ۵ · بازرسی داده — **پیش از سوزاندن ساعت‌ها GPU**
"""))

C.append(code(r"""
import glob, os, random, collections
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image

issues, areas = collections.Counter(), []
for s in ['train','val','test']:
    for lp in glob.glob(f'{DATA_DIR}/labels/{s}/*.txt'):
        for l in open(lp).read().strip().splitlines():
            t = l.split()
            if len(t) != 5: issues['بدفرمت'] += 1; continue
            c = int(float(t[0])); x,y,w,h = map(float, t[1:])
            if c not in (0,1):                 issues['کلاس نامعتبر'] += 1
            if w<=0 or h<=0:                   issues['کادر صفر'] += 1
            if min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6: issues['بیرون از کادر'] += 1
            areas.append(w*h)
print('=== اعتبار برچسب ===')
print('  ✅ هیچ مشکلی نیست' if not issues else f'  ⚠️ {dict(issues)}')
a = np.array(areas)
print(f'\nمساحت کادر: میانه {np.median(a):.4f} · p10 {np.percentile(a,10):.4f}')
print(f'سهم اشیای کوچک (زیر ۳۲×۳۲ در {IMGSZ_HI}): {(a*IMGSZ_HI*IMGSZ_HI < 32*32).mean()*100:.1f}%')
print('→ همین عدد تشخیص زودهنگام دود را تعیین می‌کند؛ imgsz را پایین‌تر نبر.')

random.seed(0)
files = random.sample(glob.glob(f'{DATA_DIR}/images/train/*'), 8)
fig, ax = plt.subplots(2, 4, figsize=(19, 8))
COL = {0:('#FF3B30','fire'), 1:('#00A8FF','smoke')}
for a_, ip in zip(ax.ravel(), files):
    im = Image.open(ip); W, H = im.size
    a_.imshow(im); a_.axis('off')
    for l in open(f"{DATA_DIR}/labels/train/{os.path.splitext(os.path.basename(ip))[0]}.txt").read().strip().splitlines():
        t = l.split()
        if len(t) != 5: continue
        c = int(t[0]); cx,cy,w,h = map(float, t[1:])
        col, nm = COL[c]
        a_.add_patch(patches.Rectangle(((cx-w/2)*W,(cy-h/2)*H), w*W, h*H, fill=False, edgecolor=col, lw=2.5))
        a_.text((cx-w/2)*W, (cy-h/2)*H-4, nm, color=col, fontsize=9, weight='bold')
plt.suptitle('🔴 = 0 fire      🔵 = 1 smoke      ← اگر جابه‌جاست، آموزش را شروع نکن', fontsize=13)
plt.tight_layout(); plt.show()
"""))

# ══════════════════════════════════════════════════════ ۶ · augmentation
C.append(md(r"""
## ۶ · augmentation مخصوص CCTV

دیتاست‌های عمومی تصویرِ **تمیزِ اینترنتی**اند. دوربین تو تصویرِ **کثیف** می‌دهد.

| افزوده | چه شکافی را می‌بندد |
|---|---|
| `ImageCompression(30..80)` | فشرده‌سازی سنگین دوربین |
| `Downscale(0.35..0.85)` | ۳۲۰×۲۴۰ تا ۷۲۰p |
| `ToGray(p=0.12)` | **حالت شبانهٔ مادون‌قرمز** |
| `MotionBlur` | لرزش و حرکت |

از **قلّابِ رسمیِ Ultralytics** استفاده می‌شود، نه دست‌کاری داخلی:
`augment.py` مدل را با `transforms=getattr(hyp, "augmentations", None)` می‌سازد و
`cfg/__init__.py` کلید `augmentations` را صراحتاً مجاز شمرده. پس با ارتقای نسخه نمی‌شکند.

> ⚠️ **این‌ها ممکن است نتیجه را بدتر کنند** — `ToGray` رنگ را از ۱۲٪ تصاویر می‌گیرد
> در حالی که نارنجی نصفِ سیگنالِ آتش است، و `Downscale` می‌تواند به اشیای کوچک
> (۲۰.۸٪ داده) آسیب بزند. برای همین **بخش ۸ آزمون A/B دارد** — حدس نمی‌زنیم.
"""))

C.append(code(r"""
CCTV_AUGS = None
if CCTV_AUG:
    try:
        import albumentations as A
        def _downscale():
            try:    return A.Downscale(scale_range=(0.35,0.85), p=0.20)
            except TypeError: return A.Downscale(scale_min=0.35, scale_max=0.85, p=0.20)
        CCTV_AUGS = [
            A.ImageCompression(quality_range=(30,80), p=0.30),
            _downscale(),
            A.ToGray(p=0.12),
            A.MotionBlur(blur_limit=5, p=0.10),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.20),
            A.CLAHE(p=0.02),
        ]
        print('✅ augmentation مخصوص CCTV آماده:')
        for t in CCTV_AUGS: print(f'   · {t.__class__.__name__:26} p={getattr(t,"p","?")}')
        print('\n   همه ImageOnly هستند → کادرها دست‌نخورده می‌مانند.')
    except Exception as e:
        print(f'⚠️ albumentations نشد ({e}) — با augmentation استاندارد ادامه می‌دهیم')
else:
    print('⏭️  خاموش است')

# ═══ تنظیمات مشترکِ آموزش — یک جا، تا مرحله‌ها با هم فرق نکنند ═══
def train_args(**over):
    a = dict(
        data=YAML, device=0, workers=WORKERS, batch=BATCH,
        project=RUNS_LOCAL, exist_ok=True, amp=True, val=True, plots=True,
        optimizer='auto', lr0=0.001, lrf=0.01, warmup_epochs=3, cos_lr=True, patience=40,
        # --- augmentation ---
        hsv_h=0.015,     # ★ کم — رنگ نارنجی نصف سیگنال آتش است
        hsv_s=0.7,
        hsv_v=0.5,       # ★ زیاد — شب و نور کم
        degrees=5.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0,
        flipud=0.0,      # آتش هرگز وارونه نیست
        fliplr=0.5,
        mosaic=1.0, close_mosaic=15, mixup=0.10, erasing=0.20,
        cache=False,     # ۱۱۳ هزار تصویر در RAM جا نمی‌شود
    )
    if CCTV_AUGS: a['augmentations'] = CCTV_AUGS
    a.update(over)
    return a
print('\n✅ train_args() آماده')
"""))

# ══════════════════════════════════════════════════════ ۷ · پرواز آزمایشی
C.append(md(r"""
## ۷ · 🚁 پرواز آزمایشی — کل خط لوله در ۳ دقیقه

**ارزشمندترین سلولِ این نوت‌بوک.**

روی ۲٪ داده و ۲ epoch اجرا می‌شود. هدف دقت نیست — هدف این است که هر خطای
پیکربندی (batch، workers، augmentation، مسیر، حافظه) **الان** پیدا شود،
نه بعد از پنج ساعت.
"""))

C.append(code(r"""
import os, glob, random, shutil, time
from ultralytics import YOLO

if RUN_SMOKETEST:
    SM = '/content/_smoke'
    shutil.rmtree(SM, ignore_errors=True)
    random.seed(0)
    for s in ['train','val']:
        os.makedirs(f'{SM}/images/{s}', exist_ok=True); os.makedirs(f'{SM}/labels/{s}', exist_ok=True)
        src = 'train' if s == 'train' else 'val'
        fs = glob.glob(f'{DATA_DIR}/images/{src}/*')
        for ip in random.sample(fs, min(len(fs), 400 if s=='train' else 120)):
            stem = os.path.splitext(os.path.basename(ip))[0]
            shutil.copy(ip, f'{SM}/images/{s}/')
            lp = f'{DATA_DIR}/labels/{src}/{stem}.txt'
            if os.path.exists(lp): shutil.copy(lp, f'{SM}/labels/{s}/')
    with open(f'{SM}/fire.yaml','w') as f:
        f.write('\n'.join([f'path: {SM}','train: images/train','val: images/val','',
                           'nc: 2','names:','  0: fire','  1: smoke','']))
    print('🚁 پرواز آزمایشی — ۲ epoch روی ~۵۲۰ تصویر ...\n')
    t = time.time()
    m = YOLO(MODEL)
    m.train(**train_args(data=f'{SM}/fire.yaml', epochs=2, imgsz=IMGSZ_LO,
                         name='_smoketest', patience=100, plots=False))
    print(f'\n✅ پرواز آزمایشی موفق در {time.time()-t:.0f} ثانیه')
    print(f'   کلاس‌های مدل: {m.model.names}  ← باید دقیقاً {{0: "fire", 1: "smoke"}} باشد')
    assert len(m.model.names) == 2, '❌ تعداد کلاس ۲ نیست!'
    print('   خط لوله سالم است. برو بخش ۹.')
    del m
    import torch, gc; gc.collect(); torch.cuda.empty_cache()
else:
    print('⏭️  پرواز آزمایشی خاموش است (توصیه نمی‌شود)')
"""))

# ══════════════════════════════════════════════════════ ۸ · خط پایه
C.append(md(r"""
## ۸ · خط پایه — مدل خام YOLO26

عددِ شروع را ثبت می‌کنیم تا «بهتر از YOLO26 خام» یک ادعای **اندازه‌گیری‌شده** باشد،
نه یک حرف.
"""))

C.append(code(r"""
from ultralytics import YOLO
import torch, time

base = YOLO(MODEL)
names = list(base.names.values())
rel = [n for n in names if 'fire' in n.lower() or 'smoke' in n.lower()]
print(f'مدل خام: {len(names)} کلاس COCO')
print(f'کلاس مرتبط با آتش/دود: {rel if rel else "❌ هیچ‌کدام → دقتش روی این مسئله صفر است"}')

dummy = torch.zeros(1,3,IMGSZ_HI,IMGSZ_HI)
for _ in range(8): base.predict(dummy, imgsz=IMGSZ_HI, verbose=False)
if torch.cuda.is_available(): torch.cuda.synchronize()
t = time.time()
for _ in range(30): base.predict(dummy, imgsz=IMGSZ_HI, verbose=False)
if torch.cuda.is_available(): torch.cuda.synchronize()
BASE_MS = (time.time()-t)/30*1000
print(f'\nتأخیر مدل خام ({MODEL}, PyTorch, imgsz={IMGSZ_HI}): {BASE_MS:.1f} ms')
print('→ در بخش ۱۲ با مدل نهایی مقایسه می‌شود.')
del base
"""))

# ══════════════════════════════════════════════════════ ۹ · A/B
C.append(md(r"""
## ۹ · 🔬 آزمون A/B روی augmentation  *(اختیاری — `RUN_ABLATION`)*

سؤالِ درست این است: **آیا augmentation مخصوص CCTV واقعاً کمک می‌کند یا ضرر می‌زند؟**

دو اجرای کوتاه و یکسان، فقط با یک تفاوت:

| اجرا | augmentation |
|---|---|
| **A** | استانداردِ Ultralytics |
| **B** | + CCTV (فشرده‌سازی · رزولوشن پایین · خاکستری · بلور) |

⚠️ **این آزمون جهت را نشان می‌دهد، نه حقیقتِ مطلق را.** روی زیرمجموعه و
epoch کم اجرا می‌شود، پس اختلافِ کمتر از ~۱ واحد mAP را جدی نگیر.
"""))

C.append(code(r"""
import os, glob, random, shutil, gc, torch
from ultralytics import YOLO

if RUN_ABLATION:
    AB = '/content/_ablation'
    if not os.path.isdir(f'{AB}/images/train'):
        shutil.rmtree(AB, ignore_errors=True); random.seed(7)
        for s, src, k in [('train','train',12000), ('val','val',1500)]:
            os.makedirs(f'{AB}/images/{s}', exist_ok=True); os.makedirs(f'{AB}/labels/{s}', exist_ok=True)
            fs = glob.glob(f'{DATA_DIR}/images/{src}/*')
            for ip in random.sample(fs, min(len(fs), k)):
                stem = os.path.splitext(os.path.basename(ip))[0]
                shutil.copy(ip, f'{AB}/images/{s}/')
                lp = f'{DATA_DIR}/labels/{src}/{stem}.txt'
                if os.path.exists(lp): shutil.copy(lp, f'{AB}/labels/{s}/')
        with open(f'{AB}/fire.yaml','w') as f:
            f.write('\n'.join([f'path: {AB}','train: images/train','val: images/val','',
                               'nc: 2','names:','  0: fire','  1: smoke','']))

    out = {}
    for tag, augs in [('A_standard', None), ('B_cctv', CCTV_AUGS)]:
        print(f'\n{"="*20} اجرای {tag} {"="*20}')
        a = train_args(data=f'{AB}/fire.yaml', epochs=15, imgsz=IMGSZ_LO,
                       name=f'_abl_{tag}', close_mosaic=3, patience=100, plots=False)
        a.pop('augmentations', None)
        if augs: a['augmentations'] = augs
        m = YOLO(MODEL); m.train(**a)
        r = m.val(data=f'{AB}/fire.yaml', imgsz=IMGSZ_LO, verbose=False)
        out[tag] = dict(map50=float(r.box.map50), map=float(r.box.map),
                        fire=float(r.box.ap50[0]), smoke=float(r.box.ap50[1]))
        del m; gc.collect(); torch.cuda.empty_cache()

    print('\n' + '='*62)
    print(f"{'اجرا':14}{'mAP@50':>11}{'mAP@50-95':>12}{'آتش':>11}{'دود':>11}")
    print('-'*62)
    for k, v in out.items():
        print(f"{k:14}{v['map50']:>11.4f}{v['map']:>12.4f}{v['fire']:>11.4f}{v['smoke']:>11.4f}")
    d = (out['B_cctv']['map50'] - out['A_standard']['map50']) * 100
    print('='*62)
    print(f'\nاختلاف (B − A) در mAP@50: {d:+.2f} واحد')
    if   d >  1.0: print('✅ CCTV کمک می‌کند — روشن نگهش دار')
    elif d < -1.0: print('❌ CCTV ضرر می‌زند — CCTV_AUG=False بگذار و بخش ۶ را دوباره اجرا کن')
    else:          print('➖ تفاوت در حدِ نویز. چون دامنهٔ واقعیِ تو CCTV است، روشن نگهش دار.')
else:
    print('⏭️  آزمون A/B خاموش است (RUN_ABLATION=False)')
"""))

# ══════════════════════════════════════════════════════ ۱۰ · آموزش
C.append(md(r"""
## ۱۰ · آموزش اصلی

### ★ رزولوشن پلکانی — چرا هم سریع‌تر است هم دقیق‌تر

| مرحله | رزولوشن | سهم epoch ها | چرا |
|---|---|---|---|
| ۱ | **۵۱۲** | ۶۰٪ | هر epoch ~۱.۵ برابر سریع‌تر؛ مدل شکل کلی شعله و دود را یاد می‌گیرد |
| ۲ | **۶۴۰** | ۴۰٪ | تثبیت روی رزولوشن نهایی؛ اشیای کوچک (۲۰.۸٪ داده) اینجا برداشت می‌شوند |

مثل یک برنامهٔ درسی: اول مفهوم، بعد جزئیات. حدود ۲۰٪ زمانِ کل صرفه‌جویی می‌شود
و mAP معمولاً برابر یا بهتر درمی‌آید.

### ⚠️ نکتهٔ فنی که خیلی‌ها را می‌سوزاند

`resume=True` در Ultralytics **همهٔ آرگومان‌ها را از چک‌پوینت بازمی‌نویسد**
(`trainer.py`: `self.args = get_cfg(ckpt_args)`).
یعنی اگر مرحلهٔ ۱ در ۵۱۲ بوده و resume کنی، در ۵۱۲ می‌ماند و مرحلهٔ ۲ هرگز اتفاق نمی‌افتد.

پس اینجا هر مرحله یک `train()` **تازه** است که از `best.pt` مرحلهٔ قبل شروع می‌کند،
و `resume` فقط **درونِ** یک مرحله استفاده می‌شود. وضعیت مرحله‌ها در
`state.json` روی درایو نگه داشته می‌شود تا قطع‌شدن Colab چیزی را خراب نکند.

### همگام‌سازی درایو در نخِ پس‌زمینه

کپی‌کردن چک‌پوینت روی درایو (FUSE) چند ثانیه طول می‌کشد. اگر در نخِ اصلی باشد،
GPU همان چند ثانیه بی‌کار می‌ماند. اینجا در یک نخِ daemon انجام می‌شود تا آموزش نایستد.
"""))

C.append(code(r"""
import os, json, shutil, glob, threading, gc, torch
from ultralytics import YOLO

STATE_F = f'{DRIVE_RUN}/state.json'
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {'done': []}
def save_state(): json.dump(state, open(STATE_F,'w'))

_lock = threading.Lock()
def make_sync(stage_name):
    def _sync(trainer):
        e = int(getattr(trainer,'epoch',0)) + 1
        if e % SYNC_EVERY and e != getattr(trainer,'epochs',0): return
        if not _lock.acquire(blocking=False): return          # کپیِ قبلی هنوز تمام نشده
        def work():
            try:
                dst = f'{DRIVE_RUN}/{stage_name}'
                os.makedirs(f'{dst}/weights', exist_ok=True)
                for w in ['last.pt','best.pt']:
                    s = f'{trainer.save_dir}/weights/{w}'
                    if os.path.exists(s): shutil.copy(s, f'{dst}/weights/{w}')
                for x in ['results.csv','args.yaml']:
                    s = f'{trainer.save_dir}/{x}'
                    if os.path.exists(s): shutil.copy(s, f'{dst}/{x}')
                print(f'   ☁️ epoch {e} → درایو')
            except Exception as ex:
                print(f'   ⚠️ همگام‌سازی: {ex}')
            finally:
                _lock.release()
        threading.Thread(target=work, daemon=True).start()
    return _sync

E1 = int(EPOCHS*STAGE1_FRAC) if PROGRESSIVE else 0
stages = ([('s1_lo', IMGSZ_LO, E1, MODEL)] if PROGRESSIVE else []) + \
         [('s2_hi', IMGSZ_HI, EPOCHS-E1, None)]

prev_best = None
for name, imgsz, eps, init in stages:
    local = f'{RUNS_LOCAL}/{RUN}_{name}'
    dbest = f'{DRIVE_RUN}/{name}/weights/best.pt'
    if name in state['done'] and os.path.exists(dbest):
        print(f'⏭️  مرحلهٔ {name} قبلاً تمام شده'); prev_best = dbest; continue

    # آیا وسط همین مرحله قطع شده؟
    dlast = f'{DRIVE_RUN}/{name}/weights/last.pt'
    resume = False
    if os.path.exists(dlast):
        os.makedirs(f'{local}/weights', exist_ok=True)
        for w in ['last.pt','best.pt']:
            s = f'{DRIVE_RUN}/{name}/weights/{w}'
            if os.path.exists(s): shutil.copy(s, f'{local}/weights/{w}')
        for x in ['results.csv','args.yaml']:
            s = f'{DRIVE_RUN}/{name}/{x}'
            if os.path.exists(s): shutil.copy(s, f'{local}/{x}')
        resume = True

    weights = f'{local}/weights/last.pt' if resume else (init or prev_best or MODEL)
    print(f'\n{"="*24} مرحلهٔ {name}  |  imgsz={imgsz}  epochs={eps} {"="*24}')
    print(f'{"🔄 ادامه از چک‌پوینت" if resume else "🆕 شروع از"}: {os.path.basename(str(weights))}')

    m = YOLO(weights)
    m.add_callback('on_fit_epoch_end', make_sync(name))
    m.train(**train_args(epochs=eps, imgsz=imgsz, name=f'{RUN}_{name}',
                         save_period=SYNC_EVERY, resume=resume))

    os.makedirs(f'{DRIVE_RUN}/{name}', exist_ok=True)
    for f in glob.glob(f'{local}/**/*', recursive=True):
        if os.path.isfile(f):
            d = os.path.join(f'{DRIVE_RUN}/{name}', os.path.relpath(f, local))
            os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy(f, d)
    state['done'].append(name); save_state()
    prev_best = f'{local}/weights/best.pt'
    del m; gc.collect(); torch.cuda.empty_cache()

BEST = prev_best
print(f'\n✅ آموزش تمام شد.\n   بهترین وزن: {BEST}\n   درایو: {DRIVE_RUN}')
"""))

# ══════════════════════════════════════════════════════ ۱۱ · ارزیابی
C.append(md(r"""
## ۱۱ · ارزیابی

**mAP دود همیشه از mAP آتش پایین‌تر می‌آید.** این عیب مدل تو نیست — مرزِ دود ذاتاً
مبهم است و دو انسان متخصص هم برای یک ستون دود کادر یکسان نمی‌کشند. جدا گزارششان کن.
"""))

C.append(code(r"""
from ultralytics import YOLO
import pandas as pd, matplotlib.pyplot as plt, os, glob

model = YOLO(BEST)
print('کلاس‌های مدل نهایی:', model.names, '  ← فقط ۲ کلاس ✓\n')
mt = model.val(data=YAML, split='test', imgsz=IMGSZ_HI, conf=0.001, iou=0.6, plots=True)

print('\n' + '='*48)
print(f"{'معیار':24}{'مقدار':>14}")
print('-'*48)
print(f"{'mAP@50':24}{mt.box.map50:>14.4f}")
print(f"{'mAP@50-95':24}{mt.box.map:>14.4f}")
print(f"{'precision':24}{mt.box.mp:>14.4f}")
print(f"{'recall':24}{mt.box.mr:>14.4f}")
print('-'*48)
for i, nm in model.names.items():
    print(f"{'mAP@50 — '+nm:24}{mt.box.ap50[i]:>14.4f}")
print('='*48)
print(f"\nمدل خام YOLO26 روی این مسئله: 0.0000  (کلاس fire/smoke نداشت)")

for name in [s[0] for s in stages]:
    csv = f'{DRIVE_RUN}/{name}/results.csv'
    if not os.path.exists(csv): continue
    df = pd.read_csv(csv); df.columns = df.columns.str.strip()
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.6))
    for c in [c for c in df.columns if 'train/' in c and 'loss' in c]:
        ax[0].plot(df['epoch'], df[c], label=c.split('/')[-1])
    ax[0].set_title(f'{name} — خطا'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    for c in ['metrics/mAP50(B)','metrics/mAP50-95(B)']:
        if c in df: ax[1].plot(df['epoch'], df[c], label=c.split('/')[-1])
    ax[1].set_title('mAP'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    for c in ['metrics/precision(B)','metrics/recall(B)']:
        if c in df: ax[2].plot(df['epoch'], df[c], label=c.split('/')[-1])
    ax[2].set_title('precision / recall'); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.show()
"""))

# ══════════════════════════════════════════════════════ ۱۲ · نقطهٔ کار
C.append(md(r"""
## ۱۲ · ★ انتخاب نقطهٔ کار بر اساس **آلارم کاذب**

اینجاست که یک مدلِ آکادمیک به یک محصول تبدیل می‌شود.

`best.pt` را Ultralytics بر اساس **fitness** انتخاب می‌کند. ولی مشتری mAP نمی‌خرد —
این را می‌پرسد:

> «شبی چند بار الکی زنگ می‌زند؟»

پس آستانه را روی **تصاویر منفی** تنظیم می‌کنیم: بالاترین recall که آلارم کاذب
هنوز زیر سقف باشد.
"""))

C.append(code(r"""
import glob, os, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO(BEST)
neg, pos = [], []
for ip in glob.glob(f'{DATA_DIR}/images/test/*'):
    lp = f"{DATA_DIR}/labels/test/{os.path.splitext(os.path.basename(ip))[0]}.txt"
    has = os.path.exists(lp) and any(l.strip() for l in open(lp).read().splitlines())
    (pos if has else neg).append(ip)
print(f'آزمون: {len(pos):,} مثبت · {len(neg):,} منفی')

SAMPLE = 2000
def max_conf(paths):
    out = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i+64], imgsz=IMGSZ_HI, conf=0.01, verbose=False):
            c = r.boxes.conf
            out.append(float(c.max()) if c is not None and len(c) else 0.0)
    return np.array(out)

cn, cp = max_conf(neg[:SAMPLE]), max_conf(pos[:SAMPLE])
print(f"\n{'آستانه':>9}{'آلارم کاذب':>15}{'تشخیص':>12}")
print('-'*38)
rows = []
for th in [0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80]:
    far, tpr = (cn>=th).mean(), (cp>=th).mean()
    rows.append((th,far,tpr)); print(f'{th:>9.2f}{far*100:>14.2f}%{tpr*100:>11.1f}%')

TARGET_FAR = 0.01
ok = [r for r in rows if r[1] <= TARGET_FAR]
if ok:
    th, far, tpr = min(ok, key=lambda r: r[0])
    print(f'\n★ آستانهٔ پیشنهادی = {th:.2f}  (آلارم کاذب {far*100:.2f}% · تشخیص {tpr*100:.1f}%)')
else:
    th = 0.5; print(f'\n⚠️ هیچ آستانه‌ای به ≤{TARGET_FAR*100:.0f}% نرسید — {th} موقتاً')
RECOMMENDED_CONF = float(th)

plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.hist(cn, bins=50, alpha=.7, label='منفی'); plt.hist(cp, bins=50, alpha=.7, label='مثبت')
plt.axvline(th, color='r', ls='--'); plt.yscale('log'); plt.legend(); plt.title('توزیع اطمینان')
plt.subplot(1,2,2)
plt.plot([r[1]*100 for r in rows],[r[2]*100 for r in rows],'o-')
plt.xlabel('آلارم کاذب %'); plt.ylabel('تشخیص %'); plt.grid(alpha=.3); plt.title('منحنی کار')
plt.tight_layout(); plt.show()
"""))

# ══════════════════════════════════════════════════════ ۱۳ · سرعت
C.append(md(r"""
## ۱۳ · خروجی صنعتی + سنجش **واقعی** سرعت

### جواب صادقانه به «آیا سریع‌تر هم می‌شویم؟»

| منبع سرعت | کمک می‌کند؟ |
|---|---|
| **خودِ fine-tune** | ❌ **هیچ** — معماری و FLOPs عوض نمی‌شود |
| سرِ ۲ کلاسه به‌جای ۸۰ | ✅ کمی — سرِ تشخیص کوچک‌تر می‌شود |
| YOLO26 بدون NMS | ✅ NMS روی CPU است و batch نمی‌شود؛ حذفش تأخیر را **قابل پیش‌بینی** می‌کند |
| `imgsz` پایین‌تر | ✅ زیاد — FLOPs با مربعِ اندازه رشد می‌کند |
| **TensorRT FP16** | ✅✅ **بیشترین برد** |

**پس بله سریع‌تر می‌شویم — ولی از مسیرِ خروجی‌گرفتن و استقرار، نه از آموزش.**
سلول زیر همه را واقعاً اندازه می‌گیرد.
"""))

C.append(code(r"""
import time, torch, os
from ultralytics import YOLO

def bench(m, imgsz, n=40, half=False):
    x = torch.zeros(1,3,imgsz,imgsz)
    for _ in range(8): m.predict(x, imgsz=imgsz, half=half, verbose=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t = time.time()
    for _ in range(n): m.predict(x, imgsz=imgsz, half=half, verbose=False)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.time()-t)/n*1000

tuned = YOLO(BEST)
rows = [('YOLO26 خام (۸۰ کلاس)', f'{IMGSZ_HI} PyTorch', BASE_MS),
        ('مدل ما (۲ کلاس)',       f'{IMGSZ_HI} PyTorch', bench(tuned, IMGSZ_HI)),
        ('مدل ما — FP16',         f'{IMGSZ_HI} PyTorch', bench(tuned, IMGSZ_HI, half=True)),
        ('مدل ما — imgsz 512',    '512 PyTorch',         bench(tuned, 512)),
        ('مدل ما — imgsz 416',    '416 PyTorch',         bench(tuned, 416))]
for fmt, lbl, kw in [('onnx','onnxruntime',dict(simplify=True)),
                     ('engine','TensorRT',dict(half=True))]:
    try:
        p = tuned.export(format=fmt, imgsz=IMGSZ_HI, **kw)
        rows.append((f'مدل ما — {fmt.upper()}', f'{IMGSZ_HI} {lbl}', bench(YOLO(p), IMGSZ_HI)))
    except Exception as e:
        print(f'⚠️ خروجی {fmt} نگرفت: {str(e)[:120]}')

print('\n' + '='*76)
print(f"{'حالت':34}{'اجرا':22}{'ms':>9}{'FPS':>10}")
print('-'*76)
for nm, mode, ms in rows: print(f'{nm:34}{mode:22}{ms:>9.1f}{1000/ms:>10.1f}')
print('='*76)
print('\nیادآوری: fine-tune سرعت را عوض نمی‌کند — خروجی و اندازهٔ ورودی عوض می‌کنند.')
"""))

# ══════════════════════════════════════════════════════ ۱۴ · ذخیره
C.append(md(r"""
## ۱۴ · ذخیره در درایو + شناسنامهٔ مدل
"""))

C.append(code(r"""
import shutil, os, glob, json

FINAL = f'{DRIVE_RUN}/final'
os.makedirs(FINAL, exist_ok=True)
shutil.copy(BEST, f'{FINAL}/best.pt')
for pat in ['*.onnx','*.engine','*.torchscript']:
    for f in glob.glob(os.path.join(os.path.dirname(BEST), pat)):
        shutil.copy(f, f'{FINAL}/{os.path.basename(f)}')

card = {
    'run': RUN, 'base_model': MODEL,
    'classes': {0:'fire', 1:'smoke'},
    'train': {'epochs': EPOCHS, 'progressive': PROGRESSIVE,
              'imgsz': [IMGSZ_LO, IMGSZ_HI] if PROGRESSIVE else [IMGSZ_HI],
              'cctv_aug': bool(CCTV_AUGS)},
    'metrics': {'mAP50': float(mt.box.map50), 'mAP50_95': float(mt.box.map),
                'mAP50_fire': float(mt.box.ap50[0]), 'mAP50_smoke': float(mt.box.ap50[1]),
                'precision': float(mt.box.mp), 'recall': float(mt.box.mr)},
    'recommended_conf': RECOMMENDED_CONF,
    'latency_ms': {nm: round(ms,2) for nm,_,ms in rows},
    'data': {'source': 'FASDD_CV (CC BY 4.0) + D-Fire',
             'traps_handled': ['کلاس برعکس D-Fire', 'تقسیم گروه‌آگاه',
                               'هم‌رویداد ۴..۱۲ بیت', 'سطل هش غول‌پیکر',
                               'سقف imgsz=640', 'resume درون‌مرحله‌ای']},
}
json.dump(card, open(f'{FINAL}/model_card.json','w'), indent=2, ensure_ascii=False)

print('✅ ذخیره شد:')
print(f'   وزن     : {FINAL}/best.pt')
print(f'   خروجی‌ها: {FINAL}/')
print(f'   شناسنامه: {FINAL}/model_card.json\n')
print(json.dumps(card, indent=2, ensure_ascii=False))
"""))

C.append(md(r"""
---

## استفاده

```python
from ultralytics import YOLO
m = YOLO('/content/drive/MyDrive/FireGuard_Runs/<RUN>/final/best.pt')
r = m.predict(frame, conf=<recommended_conf از شناسنامه>)
```

## قدم بعدی که بیشترین ارزش را دارد

~۱۲ ساعت ویدیوی خام از دوربین واقعی خودت (**بدون برچسب**) ضبط کن و بخش ۱۲ را
روی آن اجرا کن. آن‌وقت برای اولین بار عددِ «شبی چند بار الکی زنگ می‌زند» را داری —
همان عددی که مشتری می‌خرد.

بعد از آن: **BLAZE** — ردیابی + تأییدکنندهٔ سوسوی DFT + رشد ستون دود، طبق `PLAN_V2.md`.
"""))

nb = {"cells": C,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"ساخته شد: {NB}  ({len(C)} سلول, {os.path.getsize(NB)/1024:.0f} KB)")
