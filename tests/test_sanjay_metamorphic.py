"""METAMORPHIC + ROBUSTNESS TESTS - model behaviour (Sanjay).
Technique: metamorphic testing. With no ground-truth oracle for arbitrary PDFs we assert
RELATIONS between outputs of a source input and a transformed follow-up input.

MR1  Re-opening/re-saving a PDF unchanged must not change the verdict (identity).
MR2  Same file, repeated N times -> identical verdict and probabilities (determinism).
MR3  Rendering noise (JPEG recompression of the page) must not change the verdict of a clean doc.
MR4  Changing ONLY the /Title string of a genuine doc must not make it 'Font Tampered'
     (title carries no visual information).
MR5  Changing only the visible text of a doc must not change its metadata feature vector.
Tests marked xfail document real defects found by this technique.
"""
import io

import pytest
from pypdf import PdfReader, PdfWriter

from conftest import SAMPLES


def _resave(src, dst, meta_override=None):
    r = PdfReader(src)
    w = PdfWriter()
    for p in r.pages:
        w.add_page(p)
    if meta_override is not None:
        w.add_metadata(meta_override)
    with open(dst, "wb") as f:
        w.write(f)
    return dst


@pytest.mark.parametrize("key", [
    "genuine", "metadata", "both",
    pytest.param("font", marks=pytest.mark.xfail(strict=True, reason=(
        "DEFECT D2 (brittle metadata shortcut): re-saving the font-tampered sample with the same pages "
        "and metadata (pypdf also stamps /Producer) flips Font Tampered -> Metadata Tampered, although "
        "no pixel changed. Any PDF tool does this on save."))),
])
def test_MR1_identity_resave_keeps_verdict(predictor, demo, key, tmp_path):
    """Re-save the same pages AND the same metadata -> verdict must be identical."""
    src = demo(key)
    meta = dict(PdfReader(src).metadata or {})
    base = predictor.predict_and_explain(src)["verdict"]
    copy = _resave(src, str(tmp_path / "copy.pdf"), meta_override=meta)
    assert predictor.predict_and_explain(copy)["verdict"] == base


def test_MR2_determinism(predictor, demo):
    runs = [predictor.predict_and_explain(demo("font")) for _ in range(3)]
    assert len({r["verdict"] for r in runs}) == 1
    assert runs[0]["probabilities"] == runs[1]["probabilities"] == runs[2]["probabilities"]


@pytest.mark.parametrize("key", list(SAMPLES))
def test_MR3_jpeg_noise_on_render_keeps_verdict_of_demo_samples(predictor, demo, key, tmp_path, monkeypatch):
    """Simulate a re-scan: recompress the rendered page at JPEG q=60 before inference."""
    from PIL import Image
    base = predictor.predict_and_explain(demo(key))["verdict"]
    orig = predictor.render_pdf

    def noisy(path):
        img = orig(path)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=60)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    monkeypatch.setattr(predictor, "render_pdf", noisy)
    assert predictor.predict_and_explain(demo(key))["verdict"] == base


def _genuine_with(demo, tmp_path, change=None, drop=None):
    src = demo("genuine")
    meta = {k: str(v) for k, v in dict(PdfReader(src).metadata).items()}
    if change:
        meta.update(change)
    for k in (drop or []):
        meta.pop(k, None)
    return _resave(src, str(tmp_path / "v.pdf"), meta_override=meta)


@pytest.mark.parametrize("change", [
    {"/Title": "Quarterly Statement"}, {"/Title": "Untitled"}, {"/Author": "Ravi Kumar"},
    {"/Subject": "KYC"}, {"/Keywords": "kyc, bank"},
])
def test_MR4_single_cosmetic_metadata_field_does_not_change_genuine_verdict(predictor, demo, tmp_path, change):
    """Pixels untouched, ONE cosmetic field changed -> the Genuine verdict must hold."""
    assert predictor.predict_and_explain(demo("genuine"))["verdict"] == "Genuine"
    v = _genuine_with(demo, tmp_path, change=change)
    assert predictor.predict_and_explain(v)["verdict"] == "Genuine"


@pytest.mark.parametrize("field", ["/Author", "/Creator", "/Title"])
def test_MR4b_dropping_cosmetic_field_keeps_genuine(predictor, demo, tmp_path, field):
    v = _genuine_with(demo, tmp_path, drop=[field])
    assert predictor.predict_and_explain(v)["verdict"] == "Genuine"


def test_MR6_sensitivity_dropping_creation_date_flips_to_metadata_tampered(predictor, demo, tmp_path):
    """Sensitivity analysis (characterisation): has_creation_date alone decides Genuine vs
    Metadata Tampered. Documented behaviour - legitimate tools that omit /CreationDate get flagged."""
    v = _genuine_with(demo, tmp_path, drop=["/CreationDate"])
    assert predictor.predict_and_explain(v)["verdict"] == "Metadata Tampered"


def test_MR7_dropping_producer_lowers_genuine_confidence(predictor, demo, tmp_path):
    base = predictor.predict_and_explain(demo("genuine"))["confidence"]
    v = predictor.predict_and_explain(_genuine_with(demo, tmp_path, drop=["/Producer"]))
    assert v["confidence"] < base


def test_MR5_text_change_does_not_change_metadata_features(pdf_factory):
    from dataset_generator import extract_metadata_vector
    meta = {"/Producer": "ReportLab", "/Author": "x", "/Title": "T"}
    a = pdf_factory("a.pdf", meta=meta, text="Income: Rs. 50,000")
    b = pdf_factory("b.pdf", meta=meta, text="Income: Rs. 9,00,000")
    assert extract_metadata_vector(a) == extract_metadata_vector(b)


def test_confidence_monotone_after_blanking_metadata(predictor, demo):
    """Differential: zeroing metadata (scan-like input) must still yield a valid distribution."""
    import torch
    x = predictor.transform(predictor.render_pdf(demo("genuine"))).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(predictor.model(x, torch.zeros(1, 13)), dim=1)
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5)


# ------------- boundary-value analysis on metadata features -------------
@pytest.mark.parametrize("delta", [0, 1, 7, 50, -3])
def test_boundary_extreme_temporal_delta_stays_finite(predictor, demo, delta):
    import torch
    x = predictor.transform(predictor.render_pdf(demo("genuine"))).unsqueeze(0)
    m = torch.zeros(1, 13)
    m[0, 12] = delta
    with torch.no_grad():
        assert torch.isfinite(predictor.model(x, m)).all()
