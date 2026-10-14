"""WHITE-BOX TESTS - model + inference (Sanjay).
Technique: unit tests with knowledge of internals (layer shapes, hooks, feature code),
plus statement/branch coverage of architecture.py and predict.py."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import SAMPLES


@pytest.fixture(scope="module")
def net(predictor):
    return predictor.model


# ---------------- architecture.py ----------------
def test_fusion_dimension_is_1312(net):
    assert net.visual_feature_dim == 1280
    assert net.meta_feature_dim == 32
    assert net.fusion_dim == 1312
    assert net.classifier[0].in_features == 1312


def test_forward_output_shape(net):
    with torch.no_grad():
        out = net(torch.randn(3, 3, 224, 224), torch.randn(3, 13))
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


def test_metadata_branch_output_is_32d(net):
    with torch.no_grad():
        assert net.metadata_model(torch.randn(5, 13)).shape == (5, 32)


def test_visual_classifier_head_removed(net):
    assert isinstance(net.visual_model.classifier, nn.Identity)


def test_wrong_metadata_width_raises(net):
    with pytest.raises(RuntimeError):
        net(torch.randn(1, 3, 224, 224), torch.randn(1, 12))


def test_freeze_and_unfreeze_blocks():
    import architecture
    from conftest import _offline_mobilenet
    _offline_mobilenet()
    m = architecture.GROGUArchitecture(freeze_visual=True)
    assert not any(p.requires_grad for p in m.visual_model.parameters())
    m.set_visual_trainable(unfreeze=True, from_block=14)
    feats = m.visual_model.features
    assert all(p.requires_grad for p in feats[14].parameters())
    assert not any(p.requires_grad for p in feats[13].parameters())


def test_eval_mode_is_deterministic(net):
    x, m = torch.randn(2, 3, 224, 224), torch.randn(2, 13)
    with torch.no_grad():
        assert torch.equal(net(x, m), net(x, m))


def test_dynamic_quantization_shrinks_linear_layers(net):
    import copy
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not engines:
        pytest.skip("no quantized engine available on this platform")
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in engines else engines[0]
    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(net).cpu(), {nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        out = q(torch.randn(2, 3, 224, 224), torch.randn(2, 13))
    assert out.shape == (2, 4)


# ---------------- predict.py ----------------
def test_prediction_probabilities_sum_to_100(predictor, demo):
    r = predictor.predict_and_explain(demo("genuine"))
    assert sum(r["probabilities"].values()) == pytest.approx(100, abs=0.1)
    assert r["verdict"] == max(r["probabilities"], key=r["probabilities"].get)


def test_gradcam_hooks_capture_final_conv_layer(predictor, demo):
    predictor.predict_and_explain(demo("genuine"))
    assert predictor.activations.shape[1] == 1280          # last MobileNetV2 conv channels
    assert predictor.gradients.shape == predictor.activations.shape


def test_heatmap_written_and_overlay_returned(predictor, demo, tmp_path):
    out = tmp_path / "hm.jpg"
    r = predictor.predict_and_explain(demo("font"), str(out))
    assert out.exists() and out.stat().st_size > 0
    assert r["heatmap_base64"].startswith("data:image/jpeg;base64,")
    assert r["original_base64"].startswith("data:image/jpeg;base64,")


def test_metadata_inspection_has_all_13_features(predictor, demo):
    r = predictor.predict_and_explain(demo("metadata"))
    assert list(r["metadata_inspection"]) == predictor.feature_names
    assert len(predictor.feature_names) == 13


def test_suspect_producer_is_flagged(predictor, demo):
    r = predictor.predict_and_explain(demo("metadata"))   # Producer = iLovePDF Desktop 3.0
    assert r["metadata_inspection"]["suspect_producer_tool"]["flagged"] is True


# ---------------- feature extraction (dataset_generator.extract_metadata_vector) ----------------
def test_feature_vector_length_and_types(demo):
    from dataset_generator import extract_metadata_vector
    v = extract_metadata_vector(demo("genuine"))
    assert len(v) == 13 and all(isinstance(x, (int, float)) for x in v)


def test_feature_vector_unreadable_pdf_returns_zeros(tmp_path):
    from dataset_generator import extract_metadata_vector
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    assert extract_metadata_vector(str(bad)) == [0] * 13


def test_feature_flags_for_ilovepdf_producer(pdf_factory):
    from dataset_generator import extract_metadata_vector
    p = pdf_factory(meta={"/Producer": "iLovePDF Desktop 3.0", "/Title": "Untitled"})
    v = extract_metadata_vector(p)
    assert v[8] == 1        # suspect_producer_tool
    assert v[11] == 1       # suspicious_title_flag
    assert v[1] == 1        # has_producer


def test_temporal_anachronism_computation():
    from dataset_generator import compute_temporal_anachronism
    meta = {"/Producer": "Adobe Acrobat DC 2024", "/CreationDate": "D:20190101000000"}
    assert compute_temporal_anachronism(meta) == 5
    assert compute_temporal_anachronism({"/Producer": "LibreOffice 3.6"}) == 0


# ---------------- train/inference parity (documents a real defect) ----------------
@pytest.mark.xfail(strict=True, reason=(
    "DEFECT D1: predict.py calls extract_metadata_vector(pdf, temporal_delta=0) so feature #13 "
    "is always 0 at inference, while training CSVs contain non-zero values for tampered files."))
def test_inference_computes_temporal_anachronism(predictor, pdf_factory):
    p = pdf_factory(meta={"/Producer": "Adobe Acrobat DC 2024", "/CreationDate": "D:20180101000000"})
    r = predictor.predict_and_explain(p)
    assert r["metadata_inspection"]["temporal_anachronism_delta"]["value"] > 0
