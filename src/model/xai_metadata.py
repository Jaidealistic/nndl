import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Configure safe stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

FEATURE_NAMES = [
    "has_creator", "has_producer", "has_author",
    "has_creation_date", "has_mod_date",
    "creation_mod_delta_days", "creation_year_norm", "mod_year_norm",
    "suspect_producer_tool", "producer_string_length",
    "num_custom_fields", "suspicious_title_flag",
    "temporal_anachronism_delta"
]

FEATURE_LABELS = {
    "has_creator": "Creator Tag Present",
    "has_producer": "Producer Tag Present",
    "has_author": "Author Tag Present",
    "has_creation_date": "Creation Date Present",
    "has_mod_date": "Modification Date Present",
    "creation_mod_delta_days": "Creation-Mod Delta (Days)",
    "creation_year_norm": "Normalized Creation Year",
    "mod_year_norm": "Normalized Mod Year",
    "suspect_producer_tool": "Suspect Editing Tool Flag",
    "producer_string_length": "Producer String Byte-Length",
    "num_custom_fields": "Custom Metadata Fields Count",
    "suspicious_title_flag": "Suspicious Document Title",
    "temporal_anachronism_delta": "Temporal Anachronism Delta (Years)"
}

FEATURE_CATEGORIES = {
    "has_creator": "Header Provenance",
    "has_producer": "Header Provenance",
    "has_author": "Header Provenance",
    "has_creation_date": "Header Provenance",
    "has_mod_date": "Header Provenance",
    "creation_mod_delta_days": "Temporal Integrity",
    "creation_year_norm": "Temporal Integrity",
    "mod_year_norm": "Temporal Integrity",
    "suspect_producer_tool": "Software Signature",
    "producer_string_length": "Software Signature",
    "num_custom_fields": "Software Signature",
    "suspicious_title_flag": "Software Signature",
    "temporal_anachronism_delta": "Temporal Integrity"
}


class MetadataExplainer:
    """
    Student 2 Team Contribution:
    Model-agnostic and gradient-based Explainable AI (XAI) for Tabular PDF Metadata.
    Computes Input x Gradient attribution and feature perturbation impact.
    """
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES

    def explain(self, visual_tensor=None, meta_tensor=None, target_class=None, meta_grads=None, probs=None):
        """
        Computes attribution scores for each of the 13 metadata features
        towards the predicted (or target) class score.
        Reuses precomputed gradients if provided for ultra-low edge latency (<1ms).
        """
        if meta_grads is not None and meta_tensor is not None:
            grads = meta_grads[0].detach().cpu().numpy()
            values = meta_tensor[0].detach().cpu().numpy()
            attributions = grads * values
        else:
            self.model.eval()
            meta_input = meta_tensor.clone().detach().requires_grad_(True)
            visual_input = visual_tensor.clone().detach()

            logits = self.model(visual_input, meta_input)
            probs = F.softmax(logits, dim=1)
            
            if target_class is None:
                target_class = torch.argmax(probs, dim=1).item()

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

            grads = meta_input.grad[0].detach().cpu().numpy()
            values = meta_input[0].detach().cpu().numpy()
            attributions = grads * values

        # Normalize attributions to percentage contribution
        abs_attributions = np.abs(attributions)
        total_imp = np.sum(abs_attributions)
        if total_imp > 1e-7:
            normalized_attributions = (abs_attributions / total_imp) * 100.0
        else:
            normalized_attributions = np.zeros_like(abs_attributions)

        # Build feature explanation report
        explanation_list = []
        for i, name in enumerate(self.feature_names):
            val = float(values[i])
            attr = float(attributions[i])
            pct = float(normalized_attributions[i])
            cat = FEATURE_CATEGORIES.get(name, "General")
            human_label = FEATURE_LABELS.get(name, name)
            
            # Suspicion flag
            is_suspicious = False
            if name == "suspect_producer_tool" and val > 0:
                is_suspicious = True
            elif name == "temporal_anachronism_delta" and abs(val) > 0:
                is_suspicious = True
            elif name in ["has_creator", "has_producer"] and val == 0:
                is_suspicious = True

            explanation_list.append({
                "feature": name,
                "label": human_label,
                "category": cat,
                "value": round(val, 3),
                "raw_attribution": round(attr, 4),
                "importance_pct": round(pct, 2),
                "is_suspicious": is_suspicious
            })

        # Sort by importance
        explanation_list.sort(key=lambda x: x["importance_pct"], reverse=True)

        return {
            "target_class": int(target_class),
            "target_probability": round(float(probs[0, target_class].item()), 4),
            "top_features": explanation_list[:5],
            "all_features": explanation_list
        }

    def render_attribution_bar_chart(self, explanation, output_path=None):
        """Generates a clean bar chart of metadata feature attributions."""
        features = explanation["all_features"][:7] # top 7
        names = [f["label"] for f in reversed(features)]
        importances = [f["importance_pct"] for f in reversed(features)]
        colors = ["#D9534F" if f["is_suspicious"] else "#1E6B70" for f in reversed(features)]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(names, importances, color=colors, edgecolor="#333333", height=0.6)
        ax.set_xlabel("Relative Forensic Attribution (%)", fontsize=10, fontweight="bold")
        ax.set_title("Student 2 XAI: Metadata Feature Attribution (Input × Gradient)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(max(importances) * 1.25, 10))

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                    va='center', ha='left', fontsize=9, fontweight='bold')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
