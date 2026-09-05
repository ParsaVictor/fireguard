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