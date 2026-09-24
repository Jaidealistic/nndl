# GROGU: Product Modules (User Interface POV)

The GROGU interface is designed as an enterprise-grade forensic dashboard, intended for compliance officers, loan approvers, and institutional auditors. From the user's perspective, the application abstracts complex neural networks into an intuitive, zero-training-required workflow.

## Module 1: Secure Onboarding & Document Ingestion
**User POV:** 
The user is presented with a minimal, luxury-styled dashboard ("Forensic Verification."). They upload a target KYC document (PDF). 
**Functionality:** 
This module handles file parsing. It extracts the raw binary PDF file, securely rasterizes the visual layer into an image buffer (for the CNN), and simultaneously parses the hidden binary metadata headers (for the MLP). No data is sent to the cloud.

## Module 2: The Bimodal Inspection Engine (Dual-Signal Status)
**User POV:** 
Instead of a confusing "black box" percentage score, the UI displays a clear 4-class verdict (`Genuine`, `Visually Tampered`, `Metadata Tampered`, or `Both Tampered`). A dual-signal indicator shows the status of both "witnesses"—the Visual Engine and the Metadata Engine.
**Functionality:** 
If the user uploads a scanned image (which strips PDF metadata), the UI dynamically degrades, graying out the Metadata Engine and relying purely on the Visual CNN, informing the user of the reduced-confidence mode.

## Module 3: XAI (Explainable AI) Grad-CAM Viewer
**User POV:** 
If a document is flagged as "Visually Tampered", the user doesn't just have to take the machine's word for it. They click the "View Evidence" button, and the document is rendered with a spatial heat-map overlay (red hot spots).
**Functionality:** 
This module leverages Gradient-weighted Class Activation Mapping (Grad-CAM) to trace the CNN's decision back to the specific pixels (e.g., highlighting the exact forged "Monthly Income" field). This transforms an ML prediction into legally defensible, human-verifiable evidence.

## Module 4: Temporal Anachronism & Metadata Auditor
**User POV:** 
If a document is flagged as "Metadata Tampered", the user is shown a clean table highlighting the exact mathematical impossibilities found in the hidden file history.
**Functionality:** 
The UI translates raw tabular features into plain English. For example, it will highlight in red: *"ANACHRONISM DETECTED: Claimed Creation Date (2019) vs. Software Release Year (Adobe Acrobat 2024)."* This exposes sophisticated frauds that perfectly mimic fonts but fail to sanitize the PDF binary.

## Module 5: Automated Compliance Audit Report Export
**User POV:** 
With one click ("Export Audit Trail"), the user downloads a comprehensive PDF report combining the original document thumbnail, the Grad-CAM evidence heatmap, the metadata flags, and a secure cryptographic timestamp.
**Functionality:** 
This module acts as the "paper trail" for institutional compliance. If a loan is denied based on fraud, the institution has a standardized, explainable record of exactly *why* it was denied, shielding them from liability.

## Module 6: Edge-Native Offline Security Badge
**User POV:** 
A persistent badge in the UI reminds the operator: *"Edge Analytics — 100% Offline Mode."*
**Functionality:** 
Provides regulatory peace of mind. Because the model is quantized to ~3.5MB, the entire UI and backend run locally on the client machine. This mathematically guarantees compliance with strict data sovereignty laws (India's DPDP Act 2023, Europe's GDPR), as no real citizen PII is ever transmitted to a third-party API.
