"""Shared fixtures for the GROGU test suite.

Design notes
------------
* Tests run OFFLINE: architecture.py normally downloads ImageNet weights for
  MobileNetV2. The trained checkpoint overwrites every weight anyway, so we
  swap in an un-pretrained MobileNetV2 (same layer shapes) before loading it.
* The predictor / Flask app are built once per session (model load is slow).
"""
import io
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("", "src/model", "dataset"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

DEMO_DIR = os.path.join(ROOT, "demo_samples")
MODEL_PATH = os.path.join(ROOT, "src", "model", "grogu_model.pth")

SAMPLES = {
    "genuine": "sample_genuine_kyc.pdf",
    "font": "sample_font_tampered.pdf",
    "metadata": "sample_metadata_tampered.pdf",
    "both": "sample_both_tampered.pdf",
}


def _offline_mobilenet():
    import torchvision
    import architecture

    architecture.mobilenet_v2 = lambda weights=None: torchvision.models.mobilenet_v2(weights=None)


@pytest.fixture(scope="session")
def predictor():
    _offline_mobilenet()
    from predict import GROGUPredictor

    return GROGUPredictor(MODEL_PATH)


@pytest.fixture(scope="session")
def client(predictor):
    """Flask test client wired to the already-loaded predictor."""
    _offline_mobilenet()
    import app as app_module

    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture
def demo():
    """Path to a demo sample by short name: demo('genuine')."""
    return lambda key: os.path.join(DEMO_DIR, SAMPLES[key])


def make_pdf_with_meta(path, meta=None, text="Name: Test User\nIncome: Rs. 50,000"):
    """Build a small valid text PDF and (optionally) set its /Info metadata."""
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader, PdfWriter

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in text.split("\n"):
        c.drawString(60, y, line)
        y -= 20
    c.save()
    buf.seek(0)
    reader = PdfReader(buf)
    writer = PdfWriter()
    for pg in reader.pages:
        writer.add_page(pg)
    if meta is not None:
        writer.add_metadata(meta)
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    def _make(name="doc.pdf", meta=None, text="Name: Test User\nIncome: Rs. 50,000"):
        return make_pdf_with_meta(str(tmp_path / name), meta, text)

    return _make
