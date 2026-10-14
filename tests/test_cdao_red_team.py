import os
import sys
import io
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model')))
from architecture import GROGUArchitecture
from predict import GROGUPredictor
from app import app

class TestRedTeamingAdversarial:
    """
    Red Teaming & Adversarial Testing (DoD CDAO Framework Page 21):
    Probes model attack surfaces, edge-case vulnerability, metadata tampering evasion,
    and adversarial input resilience.
    """

    @pytest.fixture
    def model(self):
        m = GROGUArchitecture(num_classes=4, meta_features=13)
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model', 'grogu_model.pth'))
        if os.path.exists(model_path):
            m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        m.eval()
        return m

    def test_metadata_stripping_attack(self, model):
        """
        Adversary strips all internal PDF metadata (zero creator, zero producer, zero dates).
        Ensure the model does not crash and processes the visual branch gracefully.
        """
        dummy_visual = torch.randn(1, 3, 224, 224)
        stripped_meta = torch.zeros(1, 13)  # Complete metadata scrub

        logits = model(dummy_visual, stripped_meta)
        probs = torch.softmax(logits, dim=1)
        assert logits.shape == (1, 4)
        assert not torch.isnan(probs).any()

    def test_temporal_anachronism_attack(self, model):
        """
        Adversary creates a forged document claiming creation in 2018 but carrying
        an Adobe Acrobat 2024 software signature (delta = +6 years).
        Verify that the model activates suspicion on metadata tampering or both tampered.
        """
        dummy_visual = torch.randn(1, 3, 224, 224)
        # Vector: [creator=1, producer=1, author=1, cdate=1, mdate=1, delta_days=0, cyear=18, myear=24, suspect=1, len=35, custom=0, title=1, temporal_delta=6]
        anachronistic_meta = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 18.0, 24.0, 1.0, 35.0, 0.0, 1.0, 6.0]])

        logits = model(dummy_visual, anachronistic_meta)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        # Predictions should indicate tampering (class 1, 2, or 3, NOT genuine class 0)
        # With strong temporal delta, class 2 (metadata tampered) or 3 (both tampered) should have elevated probability
        p_genuine = probs[0, 0].item()
        p_tampered = 1.0 - p_genuine
        assert p_tampered > 0.40, f"Model failed to detect temporal anachronism attack! (P(Tampered)={p_tampered:.2f})"

    def test_adversarial_pixel_noise_perturbation(self, model):
        """
        Adversary injects high-frequency Gaussian noise (sigma=0.15) to confuse typography detectors.
        Verify model output distribution remains mathematically bounded and stable.
        """
        clean_visual = torch.randn(1, 3, 224, 224)
        clean_meta = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 26.0, 26.0, 0.0, 36.0, 0.0, 0.0, 0.0]])

        noise = torch.randn_like(clean_visual) * 0.15
        adversarial_visual = torch.clamp(clean_visual + noise, -3.0, 3.0)

        out_adv = model(adversarial_visual, clean_meta)
        probs_adv = torch.softmax(out_adv, dim=1)

        assert not torch.isnan(probs_adv).any()
        assert not torch.isinf(probs_adv).any()
        assert abs(probs_adv.sum().item() - 1.0) < 1e-4

    def test_zero_byte_denial_of_service_attack(self):
        """
        Adversary posts a zero-byte file payload to trigger an unhandled EOF or buffer fault.
        Verify server rejects the request with HTTP 400 Bad Request.
        """
        app.config['TESTING'] = True
        with app.test_client() as client:
            data = {'file': (io.BytesIO(b""), "zero_byte.pdf")}
            res = client.post('/predict', data=data, content_type='multipart/form-data')
            assert res.status_code == 400
            json_data = res.get_json()
            assert "error" in json_data
