# GROGU: Viva Preparation (Case Study Review 2 & 3)

This Q&A is designed to prepare you for rigorous technical grilling by evaluators during Review 2 and 3.

### Q1: Why did you build your own synthetic dataset instead of using real KYC documents?
**Answer:** "Real KYC documents are highly sensitive Personally Identifiable Information (PII). Handling them exposes institutions to severe legal liability under India's DPDP Act 2023 and the GDPR. Existing academic datasets like SIDTD or IDNet are flat images of identity cards. GROGU requires *digital PDF tampering* to analyze metadata. By building a pure Python, rule-based generation pipeline, we achieved 100% mathematical control over the forgeries—we know the exact pixel coordinates and byte headers of every anomaly without risking real data."

### Q2: Why did you choose MobileNetV2 instead of a heavier model like ResNet50 or Vision Transformers (ViT)?
**Answer:** "The core constraint of GROGU is that it must run offline, on edge devices, to preserve data privacy. ResNet50 has ~25 million parameters, and ViTs are even larger. MobileNetV2 uses depthwise separable convolutions, reducing the parameter count to roughly 3.4 million while maintaining extreme spatial sensitivity to font mismatches and typographical anomalies. Once quantized to INT8, it fits in a ~3.5MB footprint, allowing instant local inference without a GPU."

### Q3: Why did you use 'Late Fusion' instead of 'Early Fusion' for your bimodal architecture?
**Answer:** "Early fusion concatenates raw pixels and raw bytes into a single vector before processing. This is mathematically unstable because the variance and dimensionality of image data vastly outweighs a 13-feature tabular metadata vector. By using Late Fusion, we allow the MobileNetV2 to extract high-level semantic visual features, and the MLP to extract high-level metadata features independently. We then concatenate those processed embeddings before the final classification head. This also allows for graceful degradation (if metadata is missing, the visual branch still works)."

### Q4: Explain the 'Temporal Anachronism Delta' in your Metadata branch.
**Answer:** "Fraudsters are often meticulous about changing the visual pixels, but they fail to sanitize the hidden file history. Our pipeline calculates a 'Temporal Anachronism Delta'. If the PDF claims a `/CreationDate` of 2017, but the `/Producer` tag reveals the software used was Adobe Acrobat 2024, our engine calculates a 7-year mathematically impossible timeline. The Tabular MLP catches this instantly."

### Q5: How do you explain your model's decisions to a human auditor?
**Answer:** "We integrate XAI (Explainable AI) via Grad-CAM (Gradient-weighted Class Activation Mapping). Because MobileNetV2 is a CNN, we can extract the gradients flowing into the final convolutional layer to identify exactly which spatial regions contributed most to the 'Tampered' classification. The UI overlays a red heatmap on the original document, proving to the auditor exactly which field was forged."

### Q6: (Review 3 Scope) How will you integrate and compare with other deep learning models?
**Answer:** "For Review 3, we plan to implement a comparative baseline—an ablation study. We will compare GROGU's Bimodal architecture against a standard ResNet18 visual-only baseline. This will empirically prove that standard CNNs fail to catch 'Metadata Only' tampering, and that our fusion approach achieves significantly higher Macro F1 scores. We will also expand the UI to allow model toggling so the user can see the CNN baseline fail in real-time while the bimodal model succeeds."

### Q7: (Review 3 Scope) How does deployment to a 3rd party server work given your 'offline' constraints?
**Answer:** "While the model *can* run completely on the edge (e.g., packaged in an Electron app or WASM), enterprise deployment often requires a local microservice. We will wrap the model in a FastAPI server deployed within the client bank's secure intranet (not a public cloud). The bank's internal applications can send the PDF to `localhost:8000/predict`, ensuring the document never crosses the public internet, satisfying data sovereignty laws while utilizing standard API integration."
