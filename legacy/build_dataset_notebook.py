# -*- coding: utf-8 -*-
"""سازندهٔ نوت‌بوک آماده‌سازی دیتاست FireGuard.
نوت‌بوک را دستی ویرایش نکن — این فایل را عوض کن و دوباره اجرا کن.
    py -3 build_dataset_notebook.py
"""
import json
import os

NB = "FireGuard_Dataset.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


cells = []

# ---------------------------------------------------------------- 0
cells.append(md(r"""
# 🔥 FireGuard — آماده‌سازی دیتاست

این نوت‌بوک سه کار می‌کند و بس:

1. **دانلود** دیتاست‌ها مستقیم داخل گوگل درایو تو (فقط یک بار)
2. **ادغام** آن‌ها با اصلاح نگاشت کلاس‌ها
3. **پاک‌سازی** — حذف تکراری‌ها و تقسیم درست train/val/test

خروجی: یک پوشهٔ `fireguard_data/` آمادهٔ آموزش + فایل `fire.yaml`.

---

### ⚠️ دو تلهٔ کشف‌شده که این نوت‌بوک حلشان می‌کند

| تله | چیست | اگر حل نشود |
|---|---|---|
| **نگاشت کلاس برعکس** | FASDD: `0=fire, 1=smoke` · D-Fire: `0=smoke, 1=fire` | هر آتشی دود برچسب می‌خورد و برعکس — مدل کاملاً خراب |
| **فریم‌های پشت‌سرهم** | بخش `AoF` در D-Fire فریم‌های متوالی یک دوربین ثابت است (۸۰ تا ۹۰٪ شبیه هم) | تقسیم تصادفی → فریم مجاور در train و test → mAP دروغین بالا |

**قرارداد نهایی ما: `0 = fire` · `1 = smoke`**
"""))

# ---------------------------------------------------------------- 1
cells.append(md(r"""
## ۱ · اتصال درایو و تنظیمات

تنها سلولی که ممکن است بخواهی دست بزنی.
"""))

cells.append(code(r"""
from google.colab import drive
drive.mount('/content/drive')

import os

# --- مسیرها ---
DRIVE_DIR = '/content/drive/MyDrive/FireGuard_Datasets'   # آرشیو دائمی (zip ها)
WORK_DIR  = '/content/work'                               # فضای کار محلی (سریع)
OUT_NAME  = 'fireguard_data'                              # نام دیتاست نهایی

# --- کدام دیتاست‌ها ---
USE_FASDD  = True     # ~۹۵ هزار تصویر، ۳.۴ گیگ  — ستون فقرات
USE_DFIRE  = True     # ~۲۱.۵ هزار تصویر         — منفی‌های سخت

# --- پاک‌سازی ---
DEDUP            = True    # حذف تکراری‌ها (درون و بین دیتاست‌ها)
DEDUP_DISTANCE   = 3       # ≤ این فاصله = تکراری، حذف می‌شود
GROUP_DISTANCE   = 12      # ≤ این فاصله = «هم‌رویداد»؛ حذف نمی‌شود ولی در یک split می‌ماند
GROUP_AWARE_SPLIT = True   # فریم‌های یک رویداد را با هم در یک split نگه دار

os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
print('drive :', DRIVE_DIR)
print('work  :', WORK_DIR)
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU ندارد (برای این نوت‌بوک لازم هم نیست)'
!df -h /content | tail -1
"""))

# ---------------------------------------------------------------- 2
cells.append(md(r"""
## ۲ · دانلود داخل درایو  ← فقط بار اول

**چرا zip را باز نمی‌کنیم و در درایو نگه می‌داریم؟**

گوگل درایو با فایل‌های کوچکِ زیاد به‌شدت کند است. باز کردن ۹۵ هزار تصویر داخل درایو
ساعت‌ها طول می‌کشد و معمولاً وسط راه شکست می‌خورد. راه درست:

> **zip در درایو بماند → هر جلسه به دیسک محلی Colab کپی و باز شود (۲ تا ۳ دقیقه).**

اگر فایل از قبل باشد، دوباره دانلود نمی‌شود.
"""))

cells.append(code(r"""
!pip install -q huggingface_hub

from huggingface_hub import hf_hub_download, snapshot_download
import shutil, os, time

def human(n):
    for u in ['B','KB','MB','GB']:
        if n < 1024: return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}TB'

# ---------- FASDD_CV (3.41 GB, CC BY 4.0) ----------
if USE_FASDD:
    dst = os.path.join(DRIVE_DIR, 'FASDD_CV.zip')
    if os.path.exists(dst) and os.path.getsize(dst) > 3.3e9:
        print(f'✅ FASDD از قبل در درایو هست ({human(os.path.getsize(dst))})')
    else:
        print('⬇️  FASDD_CV.zip — ۳.۴ گیگ، حدود ۵ تا ۱۵ دقیقه ...')
        t = time.time()
        p = hf_hub_download('seawsurf/fire_smoke_dataset_fasdd_cv', 'FASDD_CV.zip',
                            repo_type='dataset', local_dir='/content/dl')
        shutil.move(p, dst)
        print(f'✅ ذخیره شد در درایو ({human(os.path.getsize(dst))}) در {time.time()-t:.0f} ثانیه')

# ---------- D-Fire (~5 GB به‌صورت فایل‌های جدا) ----------
if USE_DFIRE:
    dst = os.path.join(DRIVE_DIR, 'D-Fire.zip')
    if os.path.exists(dst) and os.path.getsize(dst) > 1e9:
        print(f'✅ D-Fire از قبل در درایو هست ({human(os.path.getsize(dst))})')
    else:
        print('⬇️  D-Fire — ۲۱.۵ هزار تصویر، حدود ۵ تا ۱۰ دقیقه ...')
        t = time.time()
        d = snapshot_download('badsaarow/d-fire', repo_type='dataset',
                              local_dir='/content/dl/dfire',
                              allow_patterns=['train/*', 'test/*'],
                              max_workers=16)
        print('   بسته‌بندی به zip ...')
        shutil.make_archive('/content/dl/D-Fire', 'zip', d)
        shutil.move('/content/dl/D-Fire.zip', dst)
        shutil.rmtree(d, ignore_errors=True)
        print(f'✅ ذخیره شد در درایو ({human(os.path.getsize(dst))}) در {time.time()-t:.0f} ثانیه')

print()
!ls -lh "$DRIVE_DIR"
"""))

# ---------------------------------------------------------------- 3
cells.append(md(r"""
## ۳ · باز کردن روی دیسک محلی

این سلول را **هر جلسه** اجرا کن (چون `/content` با قطع‌شدن Colab پاک می‌شود).
از درایو کپی می‌کند، نه دانلود دوباره از اینترنت.
"""))

cells.append(code(r"""
import os, shutil, time, zipfile

def unzip_from_drive(zip_name, out_sub):
    src = os.path.join(DRIVE_DIR, zip_name)
    out = os.path.join(WORK_DIR, out_sub)
    if os.path.isdir(out) and len(os.listdir(out)) > 0:
        print(f'✅ {out_sub} از قبل باز شده'); return out
    if not os.path.exists(src):
        print(f'⏭️  {zip_name} موجود نیست — رد شد'); return None
    print(f'📦 باز کردن {zip_name} ...')
    t = time.time()
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(out)
    print(f'   ✅ در {time.time()-t:.0f} ثانیه')
    return out

FASDD_DIR = unzip_from_drive('FASDD_CV.zip', 'fasdd') if USE_FASDD else None
DFIRE_DIR = unzip_from_drive('D-Fire.zip',  'dfire') if USE_DFIRE else None

for name, d in [('FASDD', FASDD_DIR), ('D-Fire', DFIRE_DIR)]:
    if d:
        n = sum(len(f) for _, _, f in os.walk(d))
        print(f'{name}: {n:,} فایل')
!df -h /content | tail -1
"""))

# ---------------------------------------------------------------- 4
cells.append(md(r"""
## ۴ · ادغام — با اصلاح نگاشت کلاس

**اینجا مهم‌ترین اتفاق نوت‌بوک می‌افتد.**

| منبع | کلاس در فایل اصلی | تبدیل ما |
|---|---|---|
| FASDD | `0=fire`, `1=smoke` | بدون تغییر ✅ |
| D-Fire | `0=smoke`, `1=fire` | **۰ ↔ ۱ جابه‌جا می‌شود** 🔄 |

خروجی همه‌جا: **`0 = fire` · `1 = smoke`**

تصاویر بدون شیء (منفی‌ها) با فایل برچسبِ **خالی** نگه داشته می‌شوند — این درست است و
YOLO آن‌ها را به‌عنوان پس‌زمینه یاد می‌گیرد. **دور نریزشان؛ همان‌ها FAR را پایین می‌آورند.**
"""))

cells.append(code(r"""
import os, glob, shutil, hashlib, collections, re

STAGE = os.path.join(WORK_DIR, '_stage')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

records = []   # هر رکورد: dict(src_img, label_lines, source, group, orig_split)
IMG_EXT = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')

def read_label(p):
    if not p or not os.path.exists(p): return []
    out = []
    for line in open(p, encoding='utf-8', errors='replace').read().strip().splitlines():
        t = line.split()
        if len(t) >= 5:
            out.append((int(float(t[0])), *[float(x) for x in t[1:5]]))
    return out

# ---------- FASDD : نگاشت بدون تغییر ----------
if FASDD_DIR:
    n = 0
    for split in ['train', 'valid', 'test']:
        idir = os.path.join(FASDD_DIR, split, 'images')
        if not os.path.isdir(idir): continue
        for ip in glob.glob(os.path.join(idir, '*')):
            if not ip.endswith(IMG_EXT): continue
            lp = os.path.join(FASDD_DIR, split, 'labels',
                              os.path.splitext(os.path.basename(ip))[0] + '.txt')
            base = os.path.basename(ip)
            m = re.match(r'(.+?)_jpg\.rf\.', base)
            group = m.group(1) if m else base          # شناسهٔ تصویر منبع
            records.append(dict(src=ip, lab=read_label(lp), source='fasdd',
                                group=group, split=('val' if split == 'valid' else split)))
            n += 1
    print(f'FASDD  : {n:,} تصویر  (نگاشت بدون تغییر)')

# ---------- D-Fire : جابه‌جایی ۰ ↔ ۱ ----------
if DFIRE_DIR:
    n, flipped = 0, 0
    for split in ['train', 'test']:
        idir = os.path.join(DFIRE_DIR, split, 'images')
        if not os.path.isdir(idir): continue
        for ip in glob.glob(os.path.join(idir, '*')):
            if not ip.endswith(IMG_EXT): continue
            lp = os.path.join(DFIRE_DIR, split, 'labels',
                              os.path.splitext(os.path.basename(ip))[0] + '.txt')
            raw = read_label(lp)
            fixed = [(1 - c, *rest) for (c, *rest) in raw]      # ★ 0=smoke→1 ، 1=fire→0
            flipped += len(fixed)
            stem = os.path.splitext(os.path.basename(ip))[0]
            # گروه‌بندی: AoF00123 → 'AoF001' یعنی فریم‌های نزدیک یک رویداد کنار هم می‌مانند
            m = re.match(r'([A-Za-z]+)(\d+)', stem)
            group = f'{m.group(1)}{m.group(2)[:-2]}' if m and len(m.group(2)) > 2 else stem
            records.append(dict(src=ip, lab=fixed, source='dfire',
                                group=group, split=split))
            n += 1
    print(f'D-Fire : {n:,} تصویر  ({flipped:,} کادر جابه‌جا شد ۰↔۱)  🔄')

print()
print(f'مجموع : {len(records):,} تصویر')
cnt = collections.Counter()
for r in records:
    cs = {c for c, *_ in r['lab']}
    cnt['fire' if 0 in cs else ''] += 1 if 0 in cs else 0
    cnt['smoke' if 1 in cs else ''] += 1 if 1 in cs else 0
    if not r['lab']: cnt['negative'] += 1
print(f"   دارای آتش : {cnt['fire']:,}")
print(f"   دارای دود : {cnt['smoke']:,}")
print(f"   منفی      : {cnt['negative']:,}")
"""))

# ---------------------------------------------------------------- 5
cells.append(md(r"""
## ۵ · حذف تکراری — درون و **بین** دیتاست‌ها

FASDD خودش می‌گوید منابعش شامل «دیتاست‌های آتشِ متن‌باز» است، و D-Fire یکی از
همان‌هاست. یعنی **احتمال هم‌پوشانی بین این دو جدی است**.

روش: `dHash` ۶۴ بیتی + جست‌وجوی چندشاخصی (multi-index hashing).
دو تصویر با فاصلهٔ همینگ ≤ ۳ حتماً در یکی از چهار تکهٔ ۱۶ بیتی مشترک‌اند،
پس بدون مقایسهٔ همه‌با‌همه پیدایشان می‌کنیم.
"""))

cells.append(code(r"""
import numpy as np, collections
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def dhash64(path):
    try:
        im = Image.open(path).convert('L').resize((9, 8))
        a = np.asarray(im, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        v = 0
        for b in bits: v = (v << 1) | int(b)
        return v
    except Exception:
        return None

if DEDUP:
    print(f'محاسبهٔ hash برای {len(records):,} تصویر ...')
    with ThreadPoolExecutor(16) as ex:
        hashes = list(ex.map(lambda r: dhash64(r['src']), records))
    for r, h in zip(records, hashes): r['h'] = h

    ok = [r for r in records if r['h'] is not None]
    bad = len(records) - len(ok)
    if bad: print(f'   ⚠️ {bad} تصویر خوانده نشد و حذف شد')

    parent = list(range(len(ok)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)

    # ---- گام ۱: هش دقیقاً یکسان (O(n)) --------------------------------
    # تصاویر تیره/یکنواخت هزاران‌تایی هش یکسان می‌گیرند. اگر این گام نباشد،
    # آن‌ها یک سطل غول‌پیکر می‌سازند و مقایسهٔ جفتی عملاً از کار می‌افتد.
    exact = collections.defaultdict(list)
    for i, r in enumerate(ok): exact[r['h']].append(i)
    cross_exact = 0
    for _, idxs in exact.items():
        srcs = {ok[j]['source'] for j in idxs}
        if len(srcs) > 1: cross_exact += len(idxs) - 1
        for j in idxs[1:]: union(idxs[0], j)
    uniq = [idxs[0] for idxs in exact.values()]      # فقط یک نماینده از هر هش
    print(f'   هش یکتا: {len(uniq):,} از {len(ok):,}  (هش کاملاً یکسان: {len(ok)-len(uniq):,})')

    # ---- گام ۲: نزدیک‌به‌هم، فقط روی هش‌های یکتا ------------------------
    # multi-index: دو هش با فاصلهٔ ≤3 حتماً در یکی از چهار تکهٔ ۱۶ بیتی مشترک‌اند
    NEAR = []          # جفت‌هایی که «هم‌رویداد»اند ولی تکراری نیستند
    buckets = [collections.defaultdict(list) for _ in range(4)]
    for i in uniq:
        for k in range(4):
            buckets[k][(ok[i]['h'] >> (16 * k)) & 0xFFFF].append(i)

    cross, skipped = 0, 0
    seen_pair = set()
    for k in range(4):
        for _, idxs in buckets[k].items():
            if len(idxs) < 2: continue
            if len(idxs) > 6000:            # مرز ایمنی؛ با گام ۱ عملاً هرگز نمی‌رسد
                skipped += len(idxs); continue
            for ii in range(len(idxs)):
                for jj in range(ii + 1, len(idxs)):
                    a, b = idxs[ii], idxs[jj]
                    pk = (a, b) if a < b else (b, a)
                    if pk in seen_pair: continue
                    seen_pair.add(pk)
                    d = bin(ok[a]['h'] ^ ok[b]['h']).count('1')
                    if d <= DEDUP_DISTANCE:
                        if ok[a]['source'] != ok[b]['source']: cross += 1
                        union(a, b)                      # تکراری → حذف یکی
                    elif d <= GROUP_DISTANCE:
                        NEAR.append((a, b))              # هم‌رویداد → هر دو می‌مانند
    if skipped:
        print(f'   ⚠️ {skipped:,} تصویر در سطل‌های بسیار بزرگ از مقایسهٔ جفتی جا ماند')

    groups = collections.defaultdict(list)
    for i in range(len(ok)): groups[find(i)].append(i)

    # ★ ادغام کلیدهای گروه: اگر دو تصویر تکراری بودند، گروه‌هایشان هم باید یکی شوند.
    #   (نه اینکه هر خوشه کلید تازه بگیرد — آن کار گروه‌بندی رویدادی را نابود می‌کند)
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

    kept = []
    for _, members in groups.items():
        # اگر تکراری بود، نسخه‌ای را نگه دار که برچسب بیشتری دارد
        best = max(members, key=lambda i: (len(ok[i]['lab']), ok[i]['source'] == 'fasdd'))
        kept.append(ok[best])
        base = gkey(ok[members[0]])
        for i in members:
            kunion(base, gkey(ok[i]))

    # ★ جفت‌های «هم‌رویداد» (فاصلهٔ ۴ تا ۱۲): هر دو می‌مانند، ولی باید در یک split باشند.
    #   اندازه‌گیری نشان داد فریم‌های متوالی AoF حدود ۸۰ تا ۹۰٪ شبیه‌اند، یعنی
    #   ۶ تا ۱۳ بیت اختلاف — خیلی دورتر از آستانهٔ حذف (۳)، پس فقط با نام‌گذاری
    #   گرفته نمی‌شدند و می‌توانستند بین train و test پخش شوند.
    for a, b in NEAR:
        kunion(gkey(ok[a]), gkey(ok[b]))
    print(f'   {len(NEAR):,} جفت «هم‌رویداد» پیدا شد (فاصلهٔ {DEDUP_DISTANCE+1}..{GROUP_DISTANCE}) → در یک split قفل شدند')

    removed = len(ok) - len(kept)
    print(f'✅ تکراری‌ها: {removed:,} حذف شد  ({100*removed/max(len(ok),1):.1f}%)')
    print(f'   از این تعداد، {cross + cross_exact:,} مورد بین دو دیتاست مختلف بود (هم‌پوشانی FASDD/D-Fire)')
    records = kept
else:
    KEY_PARENT = {}
    def kfind(k):
        KEY_PARENT.setdefault(k, k); return KEY_PARENT[k]
    print('⏭️  حذف تکراری خاموش است')

print(f'باقی‌مانده : {len(records):,} تصویر')
"""))

# ---------------------------------------------------------------- 6
cells.append(md(r"""
## ۶ · تقسیم گروه‌آگاه و نوشتن دیتاست نهایی

فریم‌های یک رویداد (مثلاً `AoF0403x` که ۸۰ تا ۹۰ درصد شبیه هم‌اند) **باید همه در یک
split بمانند**. وگرنه فریم شمارهٔ ۱۲ در train و فریم ۱۳ در test می‌افتد و مدل عملاً
تقلب می‌کند.
"""))

cells.append(code(r"""
import os, shutil, random, collections
from concurrent.futures import ThreadPoolExecutor

random.seed(1337)
OUT = os.path.join(WORK_DIR, OUT_NAME)
shutil.rmtree(OUT, ignore_errors=True)
for s in ['train', 'val', 'test']:
    os.makedirs(f'{OUT}/images/{s}', exist_ok=True)
    os.makedirs(f'{OUT}/labels/{s}', exist_ok=True)

if GROUP_AWARE_SPLIT:
    keyed = collections.defaultdict(list)
    for r in records:
        keyed[kfind(f"{r['source']}:{r['group']}")].append(r)
    keys = sorted(keyed)
    random.shuffle(keys)
    n = len(keys)
    bounds = (int(0.80 * n), int(0.90 * n))
    assign = {}
    for i, k in enumerate(keys):
        assign[k] = 'train' if i < bounds[0] else ('val' if i < bounds[1] else 'test')
    for k, rs in keyed.items():
        for r in rs: r['final'] = assign[k]
    print(f'تقسیم گروه‌آگاه روی {n:,} گروه')
else:
    for r in records: r['final'] = r['split'] if r['split'] in ('train','val','test') else 'train'

def write(args):
    i, r = args
    s = r['final']
    ext = os.path.splitext(r['src'])[1].lower()
    name = f"{r['source']}_{i:07d}"
    shutil.copy(r['src'], f'{OUT}/images/{s}/{name}{ext}')
    with open(f'{OUT}/labels/{s}/{name}.txt', 'w') as f:
        for c, x, y, w, h in r['lab']:
            f.write(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    return s

print('نوشتن فایل‌ها ...')
with ThreadPoolExecutor(16) as ex:
    splits = list(ex.map(write, enumerate(records)))

yaml_text = '\n'.join([
    f'path: {OUT}',
    'train: images/train',
    'val: images/val',
    'test: images/test',
    '',
    'nc: 2',
    'names:',
    '  0: fire',
    '  1: smoke',
    '',
])
with open(f'{OUT}/fire.yaml', 'w') as f:
    f.write(yaml_text)

# ---------- گزارش ----------
print()
print('=' * 58)
stat = collections.defaultdict(lambda: collections.Counter())
for r, s in zip(records, splits):
    cs = {c for c, *_ in r['lab']}
    stat[s]['images'] += 1
    if 0 in cs: stat[s]['img_fire'] += 1
    if 1 in cs: stat[s]['img_smoke'] += 1
    if not r['lab']: stat[s]['negative'] += 1
    for c, *_ in r['lab']: stat[s]['box_fire' if c == 0 else 'box_smoke'] += 1

hdr = f"{'split':6} {'تصویر':>8} {'آتش‌دار':>9} {'دوددار':>9} {'منفی':>8} {'کادر آتش':>10} {'کادر دود':>10}"
print(hdr); print('-' * 58)
for s in ['train', 'val', 'test']:
    d = stat[s]
    print(f"{s:6} {d['images']:>8,} {d['img_fire']:>9,} {d['img_smoke']:>9,} "
          f"{d['negative']:>8,} {d['box_fire']:>10,} {d['box_smoke']:>10,}")
print('=' * 58)
print(f'\n✅ دیتاست آماده: {OUT}')
print(f'   فایل پیکربندی: {OUT}/fire.yaml')

# آرشیو نهایی در درایو تا دفعهٔ بعد لازم نباشد دوباره ساخته شود
print('\n📦 بسته‌بندی نتیجه برای درایو ...')
shutil.make_archive('/content/fireguard_data', 'zip', OUT)
shutil.move('/content/fireguard_data.zip', os.path.join(DRIVE_DIR, 'fireguard_data.zip'))
print('✅ ذخیره شد:', os.path.join(DRIVE_DIR, 'fireguard_data.zip'))
"""))

# ---------------------------------------------------------------- 7
cells.append(md(r"""
## ۷ · بازبینی چشمی — **قبل از آموزش حتماً اجرا کن**

اگر کادرها روی شیء درست ننشسته‌اند، یعنی نگاشت کلاس جایی خراب شده.
🔴 قرمز = `0 fire` · 🔵 آبی = `1 smoke`
"""))

cells.append(code(r"""
import glob, random, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

random.seed(0)
files = random.sample(glob.glob(f'{OUT}/images/train/*'), 12)
fig, axes = plt.subplots(3, 4, figsize=(19, 11))
COL = {0: ('#FF3B30', 'fire'), 1: ('#00A8FF', 'smoke')}

for ax, ip in zip(axes.ravel(), files):
    im = Image.open(ip); W, H = im.size
    ax.imshow(im); ax.axis('off')
    lp = f"{OUT}/labels/train/{os.path.splitext(os.path.basename(ip))[0]}.txt"
    n = 0
    for line in open(lp).read().strip().splitlines():
        t = line.split()
        if len(t) < 5: continue
        c = int(t[0]); cx, cy, w, h = map(float, t[1:5])
        col, nm = COL[c]
        ax.add_patch(patches.Rectangle(((cx-w/2)*W, (cy-h/2)*H), w*W, h*H,
                                       fill=False, edgecolor=col, lw=2.5))
        ax.text((cx-w/2)*W, (cy-h/2)*H - 4, nm, color=col, fontsize=9, weight='bold')
        n += 1
    ax.set_title(f'{os.path.basename(ip)[:22]}  [{n}]', fontsize=8)

plt.tight_layout(); plt.show()
"""))

# ---------------------------------------------------------------- 8
cells.append(md(r"""
## بعد از این

دیتاست در `fireguard_data.zip` داخل درایو است. برای آموزش:

```python
from ultralytics import YOLO
model = YOLO('yolo26s.pt')
model.train(data=f'{OUT}/fire.yaml', epochs=120, imgsz=640, batch=16)
```

جزئیات کامل دستور پخت (augmentation، نرخ یادگیری، resume) در `DATA_AND_MODELS.md`.
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
