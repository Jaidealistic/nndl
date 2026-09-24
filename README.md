# GROGU — Generative-fraud Recognition via Optical and Genuine-metadata Unification

> **Edge-native, fully offline, dual-branch KYC document forgery detection.**
> MobileNetV2 (visual) + Tabular MLP (PDF metadata) → Late fusion → 4-class verdict + Grad-CAM heatmap

---

## 🏆 Achievements
- Shortlisted: **MeitY Cyber Security Innovation Challenge** (C-DAC Hyderabad, 2026)
- Shortlisted: **PSB Cybersecurity, Fraud & AI Hackathon** (Bank of India × IIT Hyderabad × DFS × IBA, 2026)

---

## Project Structure

```
grogu/
├── app.py                        # Flask backend API (/predict endpoint)
├── index.html                    # Interactive forensics frontend UI
├── src/model/
│   ├── architecture.py           # Bimodal model (MobileNetV2 + Tabular MLP + fusion)
│   ├── dataset.py                # PyTorch Dataset with noise augmentation
│   ├── train.py                  # Training loop with early stopping
│   ├── predict.py                # Inference engine with Grad-CAM
│   └── training_results.json     # 3-epoch PoC results (69.5% acc)
├── dataset/
│   ├── dataset_generator.py      # 10,000-sample synthetic PDF pipeline (seed=42)
│   ├── verify_dataset.py         # Dataset integrity / leakage checker
│   └── generate_samples.py       # Quick sample generation helper
├── plot_results.py               # Training curve visualiser
├── results-graph.png             # Training / validation loss & accuracy curves
├── overall-architecture-diagram.png
└── dl-architecture-diagram.png
```

---

## Architecture

```
PDF Input
   │
   ├── [Visual Branch]   MobileNetV2 → GAP → 1280-d embedding
   │
   ├── [Metadata Branch] 13 PDF attributes → FC(64)-ReLU-Dropout → 64-d embedding
   │
   └── [Late Fusion]     torch.cat(1280+64) → FC(128)-ReLU → FC(4)-Softmax
                                    ↓
              Genuine | Font Tampered | Metadata Tampered | Both
                                    +
                           Grad-CAM Heatmap Overlay
```

**Model size:** ~3.5 MB after INT8 quantisation | **Target:** Fully offline, DPDP/GDPR compliant

---

## Dataset — GROGU-KYC Bimodal Forensic Dataset

| Property | Value |
|----------|-------|
| Total documents | 10,000 PDF files |
| Classes | Genuine / Font Tampered / Metadata Tampered / Both (2,500 each) |
| Templates | 5 Indian KYC templates (Bank Statement, Salary Slip, Utility Bill, Address Proof, Income Certificate) |
| Metadata features | 13-dimensional feature vector (including temporal anachronism) |
| Train / Val / Test | 7,000 / 1,000 / 2,000 |
| Random seed | 42 (fully reproducible) |
| Real PII | ❌ None — 100% synthetic (Faker) |
| License | CC BY 4.0 |

**Dataset access:** [SharePoint](https://amritavishwavidyapeetham-my.sharepoint.com/:f:/g/personal/cb_sc_u4cse23052_cb_students_amrita_edu/IgBJziJ3yc4tTp9eiKRhsgbYAVxJBbXbNIZxaZkvzxvGeGA?e=yhp1Qy) | IEEE Dataport (pending)

---

## Quick Start

### 1. Install dependencies
```bash
pip install flask flask-cors torch torchvision opencv-python pillow pymupdf faker reportlab img2pdf pypdf pandas scikit-learn tqdm matplotlib
```

### 2. Generate the dataset
```bash
python dataset/dataset_generator.py
```

### 3. Train the model
```bash
python src/model/train.py
```

### 4. Run the forensics web app
```bash
python app.py
# Open http://127.0.0.1:5000
```

---

## Results (3-epoch CPU proof-of-concept)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Genuine | 0.697 | 1.000 | 0.821 |
| Font Tampered | 1.000 | 1.000 | **1.000** |
| Metadata Tampered | 0.510 | 0.761 | 0.610 |
| Both Tampered | 0.563 | 0.044 | 0.082 |
| **Overall Accuracy** | | | **69.5%** |

> Full 30-epoch GPU training on the 10,000-sample dataset expected to significantly improve scores (especially the "Both Tampered" class).

---

## Team

| Name | Roll No | GitHub |
|------|---------|--------|
| Sanjay AR | CB.SC.U4CSE23052 | [@sanj4git](https://github.com/sanj4git) |
| Jai Subiksha T | CB.SC.U4CSE23327 | [@Jaidealistic](https://github.com/Jaidealistic) |

**Faculty Guide:** Dr. T. Senthil Kumar, Professor, CSE Dept., Amrita Vishwa Vidyapeetham
**Course:** 23CSE473 — Neural Networks and Deep Learning
