"""SMOKE TESTS (shared) - fastest possible 'does the system stand up at all' checks.
Technique: smoke / sanity testing, black-box. Run these first after every change."""
import os

import pytest

from conftest import MODEL_PATH, SAMPLES


def test_model_checkpoint_present():
    assert os.path.isfile(MODEL_PATH)


def test_predictor_loads_and_is_in_eval_mode(predictor):
    assert predictor.model.training is False
    assert set(predictor.classes.values()) == {
        "Genuine", "Font Tampered", "Metadata Tampered", "Both Tampered"}


def test_health_endpoint_up(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "healthy"


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<html" in r.data.lower()


@pytest.mark.parametrize("key", list(SAMPLES))
def test_every_demo_sample_returns_a_verdict(client, key):
    r = client.get(f"/predict_sample/{SAMPLES[key]}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] in {"Genuine", "Font Tampered", "Metadata Tampered", "Both Tampered"}
    assert 0 <= body["confidence"] <= 100
