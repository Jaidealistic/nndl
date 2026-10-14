"""WHITE-BOX TESTS - training data loader src/model/dataset.py (Sanjay).
Technique: unit tests of GROGUBimodalDataset: tensor shapes, label mapping, noise augmentation,
missing-file fallback. Uses a tiny CSV built from the demo samples (no big dataset needed)."""
import os

import pandas as pd
import pytest
import torch
from PIL import Image

from conftest import DEMO_DIR, SAMPLES
from dataset import GROGUBimodalDataset

CLASS_OF = {"genuine": "genuine", "font": "font_tampered", "metadata": "metadata_tampered", "both": "both_tampered"}


@pytest.fixture
def csv_path(tmp_path):
    from dataset_generator import extract_metadata_vector, METADATA_FEATURE_NAMES
    rows = []
    for key, fname in SAMPLES.items():
        pdf = os.path.join(DEMO_DIR, fname)
        vec = extract_metadata_vector(pdf)
        rows.append({"pdf_path": pdf, "class": CLASS_OF[key], **dict(zip(METADATA_FEATURE_NAMES, vec))})
    rows.append({"pdf_path": str(tmp_path / "missing.pdf"), "class": "genuine",
                 **{n: None for n in METADATA_FEATURE_NAMES}})   # NaNs + missing file
    p = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_length_matches_csv(csv_path):
    assert len(GROGUBimodalDataset(csv_path)) == 5


def test_item_shapes_and_dtypes(csv_path):
    img, meta, label = GROGUBimodalDataset(csv_path)[0]
    assert img.shape == (3, 224, 224) and img.dtype == torch.float32
    assert meta.shape == (13,) and label.dtype == torch.long


def test_label_mapping(csv_path):
    ds = GROGUBimodalDataset(csv_path)
    assert [int(ds[i][2]) for i in range(4)] == [0, 1, 2, 3]


def test_nan_metadata_filled_with_zero(csv_path):
    _, meta, _ = GROGUBimodalDataset(csv_path)[4]
    assert torch.equal(meta, torch.zeros(13))


def test_missing_pdf_falls_back_to_blank_image(csv_path):
    img, _, _ = GROGUBimodalDataset(csv_path)[4]
    assert img.shape == (3, 224, 224) and torch.isfinite(img).all()


def test_noise_augmentation_changes_pixels_but_keeps_size(csv_path):
    ds = GROGUBimodalDataset(csv_path)
    clean = Image.new("RGB", (200, 200), (255, 255, 255))
    noisy = ds._apply_noise(clean)
    assert noisy.size == clean.size
    assert list(noisy.getdata()) != list(clean.getdata())


def test_unknown_class_label_defaults_to_zero(tmp_path):
    from dataset_generator import METADATA_FEATURE_NAMES
    p = tmp_path / "x.csv"
    pd.DataFrame([{"pdf_path": "nope.pdf", "class": "weird", **{n: 0 for n in METADATA_FEATURE_NAMES}}]).to_csv(p, index=False)
    assert int(GROGUBimodalDataset(str(p))[0][2]) == 0
