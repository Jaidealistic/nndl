import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model')))
from architecture import GROGUArchitecture
from xai_metadata import MetadataExplainer

class TestWhiteboxArchitecture:
    """
    White-Box Testing (DoD CDAO Framework Page 27):
    Evaluates internal model structures, dimensions, backpropagation gradient flow,
    and numerical robustness.
    """

    @pytest.fixture
    def model(self):
        m = GROGUArchitecture(num_classes=4, meta_features=13)
        m.eval()
        return m

    def test_layer_dimensions_and_concatenation(self, model):
        """Verify feature extractor output dimensions and late fusion vector size."""
        batch_size = 4
        dummy_visual = torch.randn(batch_size, 3, 224, 224)
        dummy_meta = torch.randn(batch_size, 13)

        # 1. Visual Feature Extractor
        vis_feat = model.visual_model(dummy_visual)
        assert vis_feat.shape == (batch_size, 1280), f"Expected visual features (4, 1280), got {vis_feat.shape}"

        # 2. Metadata Tabular MLP
        meta_feat = model.metadata_model(dummy_meta)
        assert meta_feat.shape == (batch_size, 32), f"Expected metadata features (4, 32), got {meta_feat.shape}"

        # 3. Full Output
        logits = model(dummy_visual, dummy_meta)
        assert logits.shape == (batch_size, 4), f"Expected logits (4, 4), got {logits.shape}"

    def test_dual_branch_gradient_flow(self):
        """Verify backpropagation simultaneously propagates non-zero gradients to both visual and tabular branches."""
        model = GROGUArchitecture(num_classes=4, meta_features=13)
        model.train()

        dummy_visual = torch.randn(2, 3, 224, 224, requires_grad=True)
        dummy_meta = torch.randn(2, 13, requires_grad=True)
        target = torch.tensor([0, 1])

        logits = model(dummy_visual, dummy_meta)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        # Check visual branch gradients
        has_vis_grad = False
        for param in model.visual_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_vis_grad = True
                break
        assert has_vis_grad, "Visual branch parameters received no gradients!"

        # Check metadata tabular branch gradients
        has_meta_grad = False
        for param in model.metadata_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_meta_grad = True
                break
        assert has_meta_grad, "Metadata branch parameters received no gradients!"

    def test_numerical_stability_and_softmax(self, model):
        """Ensure model handles zero, large, and extreme inputs without producing NaN or Inf."""
        # Test zero input
        zero_img = torch.zeros(1, 3, 224, 224)
        zero_meta = torch.zeros(1, 13)
        out_zero = model(zero_img, zero_meta)
        assert not torch.isnan(out_zero).any(), "NaN detected on zero inputs!"
        assert not torch.isinf(out_zero).any(), "Inf detected on zero inputs!"

        # Test extreme input values
        extreme_img = torch.randn(1, 3, 224, 224) * 10.0
        extreme_meta = torch.tensor([[100.0, -50.0, 1000.0, 0.0, 1.0, 5000.0, 24.0, 24.0, 1.0, 120.0, 10.0, 1.0, 15.0]])
        out_extreme = model(extreme_img, extreme_meta)
        assert not torch.isnan(out_extreme).any(), "NaN detected on extreme inputs!"
        assert not torch.isinf(out_extreme).any(), "Inf detected on extreme inputs!"

        # Test Softmax normalization
        probs = F.softmax(out_extreme, dim=1)
        prob_sum = probs.sum().item()
        assert abs(prob_sum - 1.0) < 1e-5, f"Softmax probabilities do not sum to 1.0! (Got {prob_sum})"

    def test_int8_dynamic_quantization_integrity(self, model):
        """Verify dynamic INT8 quantization converts Linear layers while maintaining valid output structure."""
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        dummy_visual = torch.randn(1, 3, 224, 224)
        dummy_meta = torch.randn(1, 13)
        out = quantized(dummy_visual, dummy_meta)
        assert out.shape == (1, 4), f"Quantized output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), "Quantized model produced NaN!"
