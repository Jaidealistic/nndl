import os
import sys
import io
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

class TestBlackboxAPI:
    """
    Black-Box Testing (DoD CDAO Framework Page 27):
    Treats the application as a black box, verifying API contracts, error boundaries,
    status codes, and input validation without internal code dependencies.
    """

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health_check_endpoint(self, client):
        """GET /health must return status healthy and 200 OK."""
        res = client.get('/health')
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("status") == "healthy"
        assert data.get("model") == "GROGU-v2.0-Bimodal"

    def test_samples_directory_endpoint(self, client):
        """GET /samples must return verified list of 4 demo KYC categories."""
        res = client.get('/samples')
        assert res.status_code == 200
        data = res.get_json()
        samples = data.get("samples", [])
        assert isinstance(samples, list)
        assert len(samples) >= 4
        ids = [s.get("id") for s in samples]
        assert "sample_genuine_kyc.pdf" in ids
        assert "sample_font_tampered.pdf" in ids
        assert "sample_metadata_tampered.pdf" in ids
        assert "sample_both_tampered.pdf" in ids

    def test_predict_sample_endpoint(self, client):
        """GET /predict_sample/sample_genuine_kyc.pdf must return 200 with complete verdict schema."""
        res = client.get('/predict_sample/sample_genuine_kyc.pdf')
        assert res.status_code == 200
        data = res.get_json()
        assert "verdict" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "metadata_inspection" in data
        assert "team_xai" in data

    def test_predict_sample_nonexistent(self, client):
        """GET /predict_sample/non_existent_id must return 404."""
        res = client.get('/predict_sample/definitely_invalid_sample_id_123.pdf')
        assert res.status_code == 404
        data = res.get_json()
        assert "error" in data

    def test_invalid_file_extension_rejection(self, client):
        """POST /predict with a .txt or .exe file must be rejected with 400 Bad Request."""
        data = {
            'file': (io.BytesIO(b"Fake malicious executable or text file"), "malicious.exe")
        }
        res = client.post('/predict', data=data, content_type='multipart/form-data')
        assert res.status_code == 400
        json_data = res.get_json()
        assert "Only PDF files are supported" in json_data.get("error", "")

    def test_corrupted_pdf_payload(self, client):
        """POST /predict with corrupted/truncated bytes must return 400 or 500 cleanly without crashing process."""
        data = {
            'file': (io.BytesIO(b"%PDF-1.4 truncated garbage content that is invalid \x00\xff"), "corrupted.pdf")
        }
        res = client.post('/predict', data=data, content_type='multipart/form-data')
        # App should cleanly handle parsing error
        assert res.status_code in [400, 500]
        json_data = res.get_json()
        assert "error" in json_data

    def test_empty_request_body(self, client):
        """POST /predict with no file payload must return 400 Bad Request."""
        res = client.post('/predict', data={})
        assert res.status_code == 400
        json_data = res.get_json()
        assert "No file uploaded" in json_data.get("error", "")
