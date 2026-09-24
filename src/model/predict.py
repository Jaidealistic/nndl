import os
import sys
import io
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import fitz

# Add dataset folder to path to import metadata extraction
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

    def predict_and_explain(self, pdf_path, output_heatmap_path):
        """Runs inference and generates Grad-CAM heatmap."""
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
        # We backpropagate the activation of the predicted class
        score = logits[0, pred_class]
        score.backward()

        # 5. Generate Grad-CAM Heatmap
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global average pooling on the gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to only keep positive influence
        cam = np.maximum(cam, 0)
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam) # Normalize
            
        # 6. Overlay Heatmap
        cam = cv2.resize(cam, original_img.size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Convert original image to BGR for OpenCV
        original_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        original_cv = np.float32(original_cv) / 255
        
        # Combine image and heatmap
        cam_result = heatmap * 0.4 + original_cv * 0.6
        cam_result = cam_result / np.max(cam_result)
        cam_result = np.uint8(255 * cam_result)
        
        cv2.imwrite(output_heatmap_path, cam_result)

        return {
            "verdict": self.classes[pred_class],
            "confidence": round(confidence * 100, 2),
            "heatmap_path": output_heatmap_path,
            "probabilities": {self.classes[i]: round(probs[0][i].item() * 100, 2) for i in range(4)}
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--model", required=True, help="Path to .pth model")
    parser.add_argument("--out", required=True, help="Path to save heatmap")
    args = parser.parse_args()
    
    predictor = GROGUPredictor(args.model)
    result = predictor.predict_and_explain(args.pdf, args.out)
    print(json.dumps(result, indent=2))
