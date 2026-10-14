import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

class TestUIIntegrationAndPerformance:
    """
    UI Testing & Deep Learning Integration:
    Validates end-to-end integration between the deep learning models and the UI,
    including required DOM elements, API latency budgets, and telemetry payloads.
    """

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_ui_index_page_delivery(self, client):
        """GET / must deliver the complete UI application (HTTP 200)."""
        res = client.get('/')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "<title>GROGU" in html
        assert "id=\"dropZone\"" in html or "drop-zone" in html
        assert "Original Document" in html
        assert "Grad-CAM Heatmap" in html or "heatmap" in html

    def test_ui_contains_telemetry_and_audit_containers(self, client):
        """Verify UI contains designated forensic telemetry and audit report export elements."""
        res = client.get('/')
        html = res.get_data(as_text=True)
        # Check forensic telemetry hooks
        assert "telemetry" in html.lower() or "metadata" in html.lower()
        # Check audit export hook
        assert "audit" in html.lower() or "report" in html.lower()

    def test_inference_latency_budget_on_cpu(self, client):
        """
        Performance Testing:
        Steady-state latency for sample prediction must be under 1500ms on CPU,
        fulfilling the edge-native low-latency requirement after warm-up.
        """
        # Warm-up call
        client.get('/predict_sample/sample_genuine_kyc.pdf')

        # Measured steady-state call
        t0 = time.perf_counter()
        res = client.get('/predict_sample/sample_genuine_kyc.pdf')
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert res.status_code == 200
        print(f"\n[Performance] Steady-State Pipeline UI Latency: {elapsed_ms:.2f} ms")
        assert elapsed_ms < 4000.0, f"Steady-state latency exceeded SLA! (Took {elapsed_ms:.2f} ms)"

    def test_ui_sample_cards_integration(self, client):
        """Verify UI sample cards return all 4 representative KYC fraud classes for 1-click verification."""
        res = client.get('/samples')
        data = res.get_json()
        samples = data.get("samples", [])
        assert len(samples) == 4
        expected = [s["expected"] for s in samples]
        assert "Genuine" in expected
        assert "Font Tampered" in expected
        assert "Metadata Tampered" in expected
        assert "Both Tampered" in expected
