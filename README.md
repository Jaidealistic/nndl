# GROGU — Deep Learning–Driven Visual and Metadata Forensics for KYC Document Fraud Detection

## Abstract
GROGU (Generative-fraud Recognition via Optical and Genuine-metadata Unification) is an edge-native, multi-modal document forensics engine designed to detect fraudulent Know-Your-Customer (KYC) documents. As digital document tampering using consumer PDF editors grows increasingly sophisticated, standard OCR-based verification tools—which evaluate *what* a document says rather than *how* it was constructed—fail to detect localized forgeries. 

To combat this, GROGU cross-examines two independent forensic signals:
1. **Visual Branch**: A MobileNetV2 CNN analyzes the pixel-level layout to detect font mismatches, typographic inconsistencies, and alignment offsets.
2. **Metadata Branch**: A Tabular Multi-Layer Perceptron (MLP) processes a 13-dimensional feature vector extracted from the hidden internal PDF file history, flagging suspicious software signatures and temporal anachronisms.

By fusing these signals, GROGU produces an explainable, four-class verdict (`Genuine`, `Font Tampered`, `Metadata Tampered`, `Both Tampered`) rather than an opaque risk score. Spatial localization via Grad-CAM further attributes the source of suspicion. Designed for 100% offline, on-device operation via INT8 post-training quantization, the entire pipeline operates with a ~3.5 MB footprint. This eliminates cloud dependencies and strict data privacy liabilities (such as India's DPDP Act 2023 and GDPR), thereby democratizing enterprise-grade forensics for smaller financial institutions.

## SDG Alignment
GROGU directly supports two core UN Sustainable Development Goals:
- **SDG 16 (Target 16.4)** — *Peace, Justice & Strong Institutions:* By detecting forged KYC documents used to open mule accounts and establish synthetic identities, GROGU strengthens institutional infrastructure against illicit financial flows and money laundering.
- **SDG 9 (Target 9.3)** — *Industry, Innovation & Infrastructure:* Traditional KYC verification relies on expensive cloud-based enterprise platforms. GROGU democratizes this forensic infrastructure, offering offline edge deployment at zero variable cost to ensure small cooperative banks, rural lenders, and fintech startups have equitable access to financial security tools.
