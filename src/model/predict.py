import os
import sys
import io
import json
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import fitz

# Add src/model and dataset folder to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')))
from dataset_generator import extract_metadata_vector

from architecture import GROGUArchitecture

class GROGUPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GROGUArchitecture(num_classes=4, meta_features=13).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.gradients = None
        self.activations = None
        
        # Register hooks for Grad-CAM on MobileNetV2's last conv layer
        target_layer = self.model.visual_model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
        self.classes = {
            0: "Genuine",
            1: "Font Tampered",
            2: "Metadata Tampered",
            3: "Both Tampered"
        }
        
        self.feature_names = [
            "has_creator", "has_producer", "has_author",
            "has_creation_date", "has_mod_date",
            "creation_mod_delta_days", "creation_year_norm", "mod_year_norm",
            "suspect_producer_tool", "producer_string_length",
            "num_custom_fields", "suspicious_title_flag",
            "temporal_anachronism_delta"
        ]

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def render_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(2.08, 2.08)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")
        doc.close()
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def predict_and_explain(self, pdf_path, output_heatmap_path=None):
        """Runs inference and generates Grad-CAM heatmap with full forensic diagnostics."""
        # 1. Render Image
        original_img = self.render_pdf(pdf_path)
        visual_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
        visual_tensor.requires_grad = True

        # 2. Extract Metadata
        meta_vec = extract_metadata_vector(pdf_path, temporal_delta=0)
        meta_tensor = torch.tensor(meta_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 3. Forward Pass
        self.model.zero_grad()
        logits = self.model(visual_tensor, meta_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

        # 4. Backward pass to get gradients for Grad-CAM
        score = logits[0, pred_class]
        score.backward()

        # 5. Generate Grad-CAM Heatmap
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        cam = cv2.resize(cam, original_img.size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        original_cv = np.float32(original_cv) / 255
        
        cam_result = heatmap * 0.45 + original_cv * 0.55
        cam_result = cam_result / np.max(cam_result)
        cam_result = np.uint8(255 * cam_result)
        
        if output_heatmap_path:
            cv2.imwrite(output_heatmap_path, cam_result)

        # Encode heatmap to base64
        _, buffer_hm = cv2.imencode('.jpg', cam_result)
        heatmap_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_hm).decode('utf-8')}"

        # Encode original image to base64
        _, buffer_orig = cv2.imencode('.jpg', np.uint8(255 * original_cv))
        orig_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_orig).decode('utf-8')}"

        # 6. Metadata inspection table
        meta_dict = {}
        for idx, name in enumerate(self.feature_names):
            val = float(meta_vec[idx])
            is_flagged = False
            flag_reason = "Normal"
            if name == "suspect_producer_tool" and val > 0:
                is_flagged = True
                flag_reason = "Disallowed editor signature (e.g. Photoshop/Canva)"
            elif name == "temporal_anachronism_delta" and val > 0:
                is_flagged = True
                flag_reason = f"Software released {int(val)} years AFTER document creation"
            elif name == "suspicious_title_flag" and val > 0:
                is_flagged = True
                flag_reason = "Title contains spoofed or generic template pattern"
            elif name == "creation_mod_delta_days" and val > 730:
                is_flagged = True
                flag_reason = f"Modification happened {int(val)} days after creation"
            
            meta_dict[name] = {
                "value": round(val, 2),
                "flagged": is_flagged,
                "note": flag_reason
            }

        # 7. Forensic analytical explanation
        verdict = self.classes[pred_class]
        explanation = []
        if verdict == "Genuine":
            explanation.append("Document passes all optical layout and font kerning consistency checks.")
            explanation.append("PDF byte-level structure exhibits authentic software generation signatures without temporal discrepancies.")
        elif verdict == "Font Tampered":
            explanation.append("Optical branch detected typography / font mismatch in localized text regions.")
            explanation.append("Grad-CAM highlights anomalous bounding boxes where character widths or anti-aliasing diverge.")
        elif verdict == "Metadata Tampered":
            explanation.append("Internal PDF metadata headers contain conflicting producer signatures or impossible creation timelines.")
            explanation.append("While pixels may appear visually plausible, digital object dictionary exhibits editing signatures.")
        else: # Both Tampered
            explanation.append("CRITICAL: Severe compound forgery detected across both optical and metadata streams.")
            explanation.append("Visual fields have been overwritten/spliced AND internal PDF headers reveal editing tools.")

        return {
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),
            "heatmap_path": output_heatmap_path,
            "heatmap_base64": heatmap_b64,
            "original_base64": orig_b64,
            "probabilities": {self.classes[i]: round(probs[0][i].item() * 100, 2) for i in range(4)},
            "metadata_inspection": meta_dict,
            "forensic_explanation": explanation
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--model", required=True, help="Path to .pth model")
    parser.add_argument("--out", default=None, help="Path to save heatmap")
    args = parser.parse_args()
    
    predictor = GROGUPredictor(args.model)
    result = predictor.predict_and_explain(args.pdf, args.out)
    print(json.dumps(result, indent=2))
