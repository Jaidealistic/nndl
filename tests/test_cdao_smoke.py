import os
import sys
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model')))
from predict import GROGUPredictor
from dataset_generator import extract_metadata_vector

class TestSmokePipeline:
    """
    Smoke Testing:
    Rapid, end-to-end sanity verification to ensure the entire operational pipeline
    functions without runtime defects.
    """

    @pytest.fixture
    def predictor(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model', 'grogu_model.pth'))
        return GROGUPredictor(model_path)

    @pytest.fixture
    def sample_pdf(self):
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo_samples', 'sample_genuine_kyc.pdf'))
        assert os.path.exists(pdf_path), f"Demo sample PDF not found at {pdf_path}"
        return pdf_path

    def test_pdf_rendering_smoke(self, predictor, sample_pdf):
        """Verify PyMuPDF renders the first PDF page into a valid PIL RGB image."""
        img = predictor.render_pdf(sample_pdf)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.width > 100 and img.height > 100

    def test_metadata_extraction_smoke(self, sample_pdf):
        """Verify 13-D metadata vector extractor produces exactly 13 numerical values."""
        vec = extract_metadata_vector(sample_pdf)
        assert len(vec) == 13
        assert all(isinstance(x, (int, float)) for x in vec)

    def test_end_to_end_predict_and_explain_smoke(self, predictor, sample_pdf):
        """Run complete prediction pipeline, ensuring Grad-CAM and Team XAI are returned."""
        result = predictor.predict_and_explain(sample_pdf)
        assert "verdict" in result
        assert result["verdict"] in ["Genuine", "Font Tampered", "Metadata Tampered", "Both Tampered"]
        assert 0.0 <= result["confidence"] <= 100.0

        # Check Heatmap base64
        assert result["heatmap_base64"].startswith("data:image/jpeg;base64,")
        assert len(result["heatmap_base64"]) > 500

        # Check Team XAI
        assert "team_xai" in result
        team_xai = result["team_xai"]
        assert "visual_xai" in team_xai
        assert "metadata_xai" in team_xai
        assert "top_contributing_features" in team_xai["metadata_xai"]
        assert len(team_xai["metadata_xai"]["top_contributing_features"]) > 0
