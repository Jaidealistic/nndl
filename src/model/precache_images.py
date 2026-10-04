import os
import sys
import time
import pandas as pd
import fitz
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'grogu_dataset', 'cache')

def render_single_pdf(pdf_path):
    try:
        abs_pdf = os.path.join(PROJECT_ROOT, pdf_path) if not os.path.isabs(pdf_path) else pdf_path
        if not os.path.exists(abs_pdf):
            return False
        
        base_name = os.path.basename(pdf_path)
        out_path = os.path.join(CACHE_DIR, f"{base_name}.jpg")
        
        if os.path.exists(out_path):
            return True
        
        doc = fitz.open(abs_pdf)
        page = doc[0]
        # Render at 150 DPI
        mat = fitz.Matrix(2.08, 2.08)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        
        # Save as optimized JPEG (224x224 directly to avoid resizing during training)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        img.save(out_path, format="JPEG", quality=85)
        doc.close()
        return True
    except Exception as e:
        return False

def precache_split(csv_path, max_workers=14):
    df = pd.read_csv(csv_path)
    paths = df['pdf_path'].tolist()
    total = len(paths)
    print(f"Precaching {total} files from {os.path.basename(csv_path)} using {max_workers} processes...")
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(render_single_pdf, paths, chunksize=25))
    t1 = time.time()
    
    success = sum(1 for r in results if r)
    elapsed = t1 - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"Cached {success}/{total} in {elapsed:.2f}s ({rate:.1f} docs/sec)")

if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    train_csv = os.path.join(PROJECT_ROOT, 'grogu_dataset', 'splits', 'train.csv')
    val_csv   = os.path.join(PROJECT_ROOT, 'grogu_dataset', 'splits', 'val.csv')
    test_csv  = os.path.join(PROJECT_ROOT, 'grogu_dataset', 'splits', 'test.csv')
    
    print("Starting GROGU Dataset Pre-Caching...")
    precache_split(val_csv)
    precache_split(test_csv)
    precache_split(train_csv)
    print("Pre-caching complete!")
