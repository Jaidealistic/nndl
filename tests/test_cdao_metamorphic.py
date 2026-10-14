import os
import sys
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model')))
from architecture import GROGUArchitecture
from predict import GROGUPredictor

class TestMetamorphicRelations:
    """
    Metamorphic Testing (DoD CDAO Framework Page 22):
    Evaluates Metamorphic Relations (MR) where input transformations should preserve
    specific expected behavioral invariants without needing an external oracle.
    """

    @pytest.fixture
    def predictor(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model', 'grogu_model.pth'))
        return GROGUPredictor(model_path)

    @pytest.fixture
    def sample_pdf(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo_samples', 'sample_genuine_kyc.pdf'))

    def test_mr1_contrast_invariance(self, predictor, sample_pdf):
        """
        MR1 (Visual Contrast Invariance):
        Slight contrast adjustments (+/- 5%) simulating scanning brightness variations
        should maintain consistent model output.
        """
        orig_img = predictor.render_pdf(sample_pdf)
        
        # Apply +5% and -5% contrast
        enhancer = ImageEnhance.Contrast(orig_img)
        img_high = enhancer.enhance(1.05)
        img_low = enhancer.enhance(0.95)

        t_orig = predictor.transform(orig_img).unsqueeze(0)
        t_high = predictor.transform(img_high).unsqueeze(0)
        t_low = predictor.transform(img_low).unsqueeze(0)

        dummy_meta = torch.zeros(1, 13)

        with torch.no_grad():
            out_orig = torch.softmax(predictor.model(t_orig, dummy_meta), dim=1)
            out_high = torch.softmax(predictor.model(t_high, dummy_meta), dim=1)
            out_low = torch.softmax(predictor.model(t_low, dummy_meta), dim=1)

        pred_orig = torch.argmax(out_orig, dim=1).item()
        pred_high = torch.argmax(out_high, dim=1).item()
        pred_low = torch.argmax(out_low, dim=1).item()

        # Both transformed versions should match the baseline class
        assert pred_orig == pred_high == pred_low, (
            f"MR1 violated: contrast shift flipped prediction ({pred_orig} -> high:{pred_high}, low:{pred_low})"
        )

    def test_mr2_benign_metadata_invariance(self, predictor):
        """
        MR2 (Benign Metadata Modification Invariance):
        Adding a neutral metadata field (e.g. increasing producer string length slightly)
        without suspect keywords should not flip an authentic prediction into fraudulent.
        """
        dummy_visual = torch.randn(1, 3, 224, 224)
        
        # Baseline genuine vector: no suspect tools, zero temporal delta
        meta_base = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 26.0, 26.0, 0.0, 30.0, 0.0, 0.0, 0.0]])
        # Transformed vector: slightly different benign length (30 -> 35), still 0 suspect, 0 temporal
        meta_trans = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 26.0, 26.0, 0.0, 35.0, 0.0, 0.0, 0.0]])

        with torch.no_grad():
            out_base = torch.softmax(predictor.model(dummy_visual, meta_base), dim=1)
            out_trans = torch.softmax(predictor.model(dummy_visual, meta_trans), dim=1)

        diff = torch.abs(out_base - out_trans).max().item()
        assert diff < 0.15, f"MR2 violated: benign metadata adjustment caused probability jump of {diff:.3f}"

    def test_mr3_dpi_scaling_invariance(self, predictor, sample_pdf):
        """
        MR3 (Rendering DPI Invariance):
        Rendering at 150 DPI vs 300 DPI must both be successfully normalized by
        torchvision transforms to (224, 224) without dimension or stride divergence.
        """
        import fitz
        doc = fitz.open(sample_pdf)
        page = doc[0]

        # 150 DPI (matrix 1.5)
        pix_150 = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_150 = Image.frombytes("RGB", [pix_150.width, pix_150.height], pix_150.samples)

        # 300 DPI (matrix 3.0)
        pix_300 = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
        img_300 = Image.frombytes("RGB", [pix_300.width, pix_300.height], pix_300.samples)
        doc.close()

        t_150 = predictor.transform(img_150).unsqueeze(0)
        t_300 = predictor.transform(img_300).unsqueeze(0)

        assert t_150.shape == (1, 3, 224, 224)
        assert t_300.shape == (1, 3, 224, 224)

        with torch.no_grad():
            dummy_meta = torch.zeros(1, 13)
            out_150 = torch.softmax(predictor.model(t_150, dummy_meta), dim=1)
            out_300 = torch.softmax(predictor.model(t_300, dummy_meta), dim=1)

        p1 = torch.argmax(out_150, dim=1).item()
        p2 = torch.argmax(out_300, dim=1).item()
        assert p1 == p2, f"MR3 violated: DPI scaling flipped prediction ({p1} != {p2})"
