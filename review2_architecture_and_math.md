# GROGU: Deep Learning Architecture & Mathematical Foundations

This document provides a comprehensive overview of the deep learning architecture chosen for the GROGU engine, the mathematical foundations of its components, and a rigorous justification of its hyperparameters.

## 1. Architectural Theory: The Bimodal Forgery Engine

GROGU employs a **Late Fusion Bimodal Neural Network** consisting of two independent processing branches that merge at the penultimate layer. This architecture was explicitly chosen to address the two vectors of digital document forgery: Pixel-level (Visual) tampering and Byte-level (Metadata) tampering.

### A. The Visual Branch (MobileNetV2 CNN)
We employ **MobileNetV2** as the visual feature extractor. 
*   **Why MobileNetV2?** Standard OCR models (like Tesseract/TrOCR) read *what* the text says, making them blind to font mismatches. Heavy vision models (ResNet50, ViT) require significant compute and memory. MobileNetV2 is optimized for edge devices, using depthwise separable convolutions that drastically reduce the parameter count (to ~3.4M) while retaining the capacity to detect micro-anomalies in texture, font aliasing, and spatial alignment.
*   **Transfer Learning:** The network is pre-trained on ImageNet to learn robust primitive edge and texture filters, then fine-tuned on our synthetic document dataset. The final classification head is stripped, extracting a 1280-dimensional feature vector.

### B. The Metadata Branch (Tabular MLP)
Digital forgeries using consumer software (e.g., Adobe Acrobat, iLovePDF) inevitably leave structural metadata traces.
*   We use a **Multi-Layer Perceptron (MLP)**.
*   The input is a 13-dimensional handcrafted feature vector extracted from the PDF binary (e.g., temporal anachronism deltas, missing author fields).
*   The MLP projects this low-dimensional data into a 32-dimensional dense representation, mapping non-linear relationships (e.g., a missing `/Author` tag is benign, but a missing `/Author` *combined* with a 3-year `temporal_anachronism_delta` strongly correlates with fraud).

### C. Late Fusion & Classification
The 1280-D visual vector and 32-D metadata vector are concatenated into a 1312-D vector. 
*   **Why Late Fusion?** Early fusion (combining raw pixels and bytes) is mathematically unstable due to massive modality variance. Late fusion allows each branch to independently form high-level semantic conclusions before merging. If the document is a scanned image (metadata branch yields 0s), the visual branch can still drive the final classification independently (graceful degradation).

---

## 2. Mathematical Formalization

### A. Depthwise Separable Convolution (Visual Branch)
Standard convolutions apply filters across all spatial and channel dimensions simultaneously, resulting in computational cost $O(D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F)$.
MobileNetV2 splits this into two steps:
1.  **Depthwise Convolution:** Applies a single spatial filter per input channel.
    $$ \hat{G}_{k, l, m} = \sum_{i,j} \hat{K}_{i,j,m} \cdot F_{k+i-1, l+j-1, m} $$
2.  **Pointwise Convolution:** Applies a $1 \times 1$ convolution to linearly combine the channels.

This reduces the computational cost by a factor of $\frac{1}{N} + \frac{1}{D_K^2}$, making edge-deployment feasible.

### B. Late Fusion Representation
Let $X_v$ be the visual input and $X_m$ be the metadata vector.
The visual feature vector is $f_v(X_v) \in \mathbb{R}^{1280}$.
The metadata feature vector is $f_m(X_m) \in \mathbb{R}^{32}$.
The fused representation $Z$ is the concatenation:
$$ Z = [ f_v(X_v) \parallel f_m(X_m) ] \in \mathbb{R}^{1312} $$

The final class probability $\hat{y}$ over classes $C = 4$ is computed via Softmax:
$$ P(y = c \mid Z) = \frac{\exp(W_c^T Z + b_c)}{\sum_{k=1}^{K} \exp(W_k^T Z + b_k)} $$

### C. Temporal Anachronism Delta ($f_{13}$)
A core feature of the metadata vector is the mathematical proof of an impossible timeline.
$$ \Delta_{temporal} = \max(0, Y_{software\_release} - Y_{claimed\_creation}) $$
If $\Delta_{temporal} > 0$, the document was edited using software that did not exist when the document was supposedly created.

---

## 3. Hyperparameters & Justification

| Hyperparameter | Value | Justification |
| :--- | :--- | :--- |
| **Batch Size** | 32 | Optimizes GPU memory while providing a sufficiently accurate gradient estimate. Small batches help regularize the network (avoiding sharp minima). |
| **Learning Rate** | 1e-4 | A conservative learning rate prevents catastrophic forgetting of the pre-trained ImageNet weights in the MobileNetV2 backbone. |
| **Optimizer** | Adam | Adaptive moment estimation converges significantly faster than standard SGD on sparse, high-variance gradients produced by the tabular MLP branch. |
| **Weight Decay** | 1e-5 | $L_2$ regularization penalty to prevent overfitting on the synthetic training set, forcing the network to learn generalized font anomalies rather than memorizing specific pixel noise. |
| **Dropout** | 0.3 / 0.4 | Applied heavily (30% in MLP, 40% in Fusion head) to force the network to rely on both visual and metadata signals, preventing co-adaptation where the network simply ignores one branch. |
| **Loss Function** | Cross-Entropy | Standard mathematical objective for multi-class classification, heavily penalizing confident but incorrect predictions. |
| **Quantization Precision**| INT8 | Post-training dynamic quantization maps FP32 weights to 8-bit integers. Reduces model size from ~14MB to ~3.5MB for offline edge-deployment with negligible accuracy drop. |
