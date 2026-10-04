import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import io
import cv2
import numpy as np
import random

# Try to import fitz (PyMuPDF) for on-the-fly PDF rendering
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

class GROGUBimodalDataset(Dataset):
    """
    GROGU Bimodal Dataset.
    Reads directly from a splits CSV which contains:
    - pdf_path: path to raw PDF
    - class: class label string
    - All 13 metadata features inline
    
    Renders PDF to image on-the-fly using PyMuPDF.
    Falls back to blank white image if PDF can't be read.
    """
    
    META_FEATURES = [
        "has_creator", "has_producer", "has_author",
        "has_creation_date", "has_mod_date",
        "creation_mod_delta_days", "creation_year_norm", "mod_year_norm",
        "suspect_producer_tool", "producer_string_length",
        "num_custom_fields", "suspicious_title_flag",
        "temporal_anachronism_delta"
    ]
    
    CLASS_MAP = {
        "genuine": 0,
        "font_tampered": 1,
        "metadata_tampered": 2,
        "both_tampered": 3
    }

    def __init__(self, split_csv_path, transform=None):
        self.df = pd.read_csv(split_csv_path)
        
        # Fill NaN metadata values with 0
        self.df[self.META_FEATURES] = self.df[self.META_FEATURES].fillna(0)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _render_pdf_to_image(self, pdf_path):
        """Render first page of PDF to PIL Image using cache if available, else PyMuPDF."""
        # 1. Check if cached JPEG exists
        base_name = os.path.basename(pdf_path)
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'grogu_dataset', 'cache')
        cached_img_path = os.path.join(cache_dir, f"{base_name}.jpg")
        if os.path.exists(cached_img_path):
            try:
                return Image.open(cached_img_path).convert("RGB")
            except Exception:
                pass

        # 2. Path resolution
        if not os.path.isabs(pdf_path):
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            abs_path = os.path.join(proj_root, pdf_path)
        else:
            abs_path = pdf_path

        # 3. Fallback to direct PyMuPDF rendering
        if HAS_FITZ and os.path.exists(abs_path):
            try:
                doc = fitz.open(abs_path)
                page = doc[0]
                mat = fitz.Matrix(2.08, 2.08)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                return img
            except Exception:
                pass
        # Fallback: blank white image
        return Image.new("RGB", (224, 224), (255, 255, 255))

    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Inject synthetic noise to simulate real-world scanned documents."""
        # 1. Convert PIL to CV2 BGR
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 2. JPEG Compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 70)]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
        
        # 3. Gaussian Blur (simulate out of focus)
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
        # 4. Gaussian Noise
        row, col, ch = img.shape
        mean = 0
        sigma = random.uniform(10, 25)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        img = np.clip(noisy, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Visual: render PDF to image
        image = self._render_pdf_to_image(str(row["pdf_path"]))
        image = self._apply_noise(image)
        image_tensor = self.transform(image)

        # 2. Metadata: read 13 features directly from CSV
        meta = row[self.META_FEATURES].astype(float).values
        meta_tensor = torch.tensor(meta, dtype=torch.float32)

        # 3. Label
        label = self.CLASS_MAP.get(str(row["class"]), 0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, meta_tensor, label_tensor
