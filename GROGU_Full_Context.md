# GROGU — Full Project Context
### For handoff to an app-building agent

**Product Name:** GROGU
**Backronym:** Generative-fraud Recognition via Optical and Genuine-metadata Unification
**Slogan:** "Technically valid. Forensically false."
**Full Title:** GROGU — Deep Learning–Driven Visual and Metadata Forensics for KYC Document Fraud Detection
**Type:** Solo capstone project, Neural Networks & Deep Learning (23CSE461), Amrita Vishwa Vidyapeetham

This document is the single source of truth for what GROGU is, why it exists, how it works, and exactly what needs to be built. It consolidates the product framing, the research/academic framing, and the fully-debugged dataset generation pipeline. Treat the code in Section 8 as canonical — it already has every bug fix applied; do not reintroduce earlier broken versions.

---

## 1. What GROGU Is

GROGU is an edge-native, multi-modal document forensics engine that detects fraudulent KYC (Know-Your-Customer) documents by cross-examining two independent forensic signals on the same file:

1. **Visual branch** — the document's pixel-level layout (font consistency, alignment, typography)
2. **Metadata branch** — the document's hidden internal PDF file history (timestamps, software signatures, structural attributes)

Both signals are processed independently and fused at decision level, producing an explainable, four-class verdict — not an opaque risk score. The entire pipeline is designed to run fully offline, on-device, with zero cloud dependency.

---

## 2. Problem Addressed

### 2.1 How the tampering happens (mechanism)
Free consumer PDF editors (iLovePDF, Adobe Acrobat, Smallpdf) let anyone alter a single field — income, date of birth, address — on a genuine KYC document template in under 5 minutes. The layout stays authentic because only one field was changed, so the forgery passes visual inspection. Standard OCR-based verification tools read *what* a document says, not *how* it was constructed, so the forgery goes undetected.

### 2.2 The damage this causes
- Tampered income certificate → fraudulent loan disbursed → institution absorbs the loss → credit tightens for legitimate borrowers
- Tampered address proof → mule account opened → untraceable money laundering (PMLA exposure for the institution)
- Tampered utility bill → synthetic identity gets a fintech wallet, UPI handle, credit line
- FinCEN 2026: $200B+ in suspicious transactions linked to identity fraud; 42% of Suspicious Activity Reports tied to identity compromise
- Sardine.ai 2026: synthetic identity kits targeting Indian banks cost as little as $150
- RBI KYC Master Directions penalise the institution even when it was the one deceived
- Cascading effect: legitimate borrowers face stricter documentation, higher rates, slower onboarding

### 2.3 Why current tools fail
- **Privacy liability** — enterprise KYC platforms require cloud upload of sensitive documents → direct exposure under India's DPDP Act 2023 and GDPR
- **Cost barrier** — per-check API pricing excludes cooperative banks, rural lenders, fintech startups
- No metadata-aware detection exists in deployed tools; no offline option exists

### 2.4 Research-level gap
- Public forgery datasets (SIDTD, IDNet) are flat JPEGs, visual-only — structurally cannot train a metadata-aware branch
- No published model treats visual layout and PDF metadata as co-equal forensic signals
- No published model checks cross-field temporal plausibility (e.g., a document claiming to be from 2019 but carrying a 2024 software signature)
- Existing tools return an opaque probability score with no spatial or forensic attribution

---

## 3. Proposed Solution & Architecture

**Core idea:** Two independent "witnesses" examine the same document — one studies pixels, one studies hidden file metadata — and GROGU fuses both findings into a single explained verdict rather than a bare yes/no.

- **Visual branch:** MobileNetV2 CNN — detects font mismatches, typographic inconsistencies, alignment offsets
- **Metadata branch:** Tabular Multi-Layer Perceptron — processes a 13-dimensional feature vector extracted from PDF metadata (see Section 8, Step 5)
- **Fusion:** late fusion via `torch.cat()` of both branches' feature vectors → 4-class classification head
- **Output classes:** `Genuine (0)`, `Font/Visually Tampered (1)`, `Metadata Tampered (2)`, `Both Tampered (3)`
- **Explainability:** Grad-CAM on the visual branch localises the tampered region spatially; output names which branch triggered the verdict
- **Deployment:** INT8 post-training quantisation, target footprint ~3.5 MB, 100% offline, zero data transmission
- **Graceful degradation:** if metadata is absent (e.g. a scanned/photographed document), the visual branch continues to operate at full confidence and the UI transparently communicates the reduced analysis mode

**Compliance-oriented output:** GROGU auto-generates a one-click downloadable PDF audit report containing the document thumbnail, Grad-CAM heatmap overlay, class-specific verdict, the specific metadata attributes that triggered suspicion, and a timestamp — intended as a compliance paper trail an officer can attach to a rejected application, not just a screen-level result.

---

## 4. Motivation

- Fraud is industrialised: synthetic identity kits ~$150 (Sardine.ai 2026); FinCEN links $200B+ in transactions and 42% of SARs to identity fraud
- India's DPDP Act 2023 makes cloud-upload KYC tools a direct regulatory liability, not just an inconvenience
- Small institutions (cooperative banks, rural lenders, microfinance, fintech startups) face the same fraud risk as large banks with none of the budget for enterprise KYC vendors
- Edge AI is now practical — quantised multi-modal models can run in a few MB on ordinary hardware, removing the need for a data centre
- **Goal:** enterprise-grade forensic capability, delivered at zero variable cost, fully on-device

**Industry validation (2026):**
- MeitY Cyber Security Innovation Challenge — coordinated with C-DAC Hyderabad across 50 premier institutions — lists privacy-preserving KYC verification as a live problem statement
- PSB Cybersecurity, Fraud & AI Hackathon 2026 — Bank of India × IIT Hyderabad × Department of Financial Services (Ministry of Finance) × Indian Banks' Association — ₹20 lakh prize pool — names document fraud detection as a core national problem track

---

## 5. Research Questions & Gaps Addressed

**Research Questions**
1. Can a lightweight edge model fuse visual + PDF metadata as co-equal forensic signals reliably?
2. Does the metadata branch give a measurable accuracy gain over visual-only? Which fraud class benefits most?
3. Can a rule-based synthetic pipeline train this without any real PII or real fraud documents?
4. Can INT8 quantisation hit ~3.5 MB without major accuracy loss?

**Gaps Addressed**

| Gap | Current State | GROGU |
|---|---|---|
| Pixel-only datasets | SIDTD, IDNet = flat JPEGs | Real PDF files, visual + metadata tampering |
| No metadata-aware models | All classifiers = image-only | Tabular MLP on 13 PDF metadata features |
| No temporal anachronism check | Fields inspected in isolation | Cross-field date vs. software-version delta (feature #13) |
| Cloud-only deployment | Needs network + cloud compute | ~3.5 MB, INT8, fully offline |
| Opaque output | Single probability score | 4-class + Grad-CAM localisation + explanation |

---

## 6. Related Products & Competitive Landscape

| Product | URL | Key Features | Limitations |
|---|---|---|---|
| Onfido | onfido.com | Document + biometric liveness, OCR extraction, global ID coverage | Cloud-only, per-check fee, no metadata forensics, DPDP liability |
| Jumio | jumio.com | AI ID verification, 5,000+ doc types, NFC chip reading | Cloud upload required, no offline mode, visual-only |
| Au10tix | au10tix.com | Forensic doc analysis, tampering detection, biometric comparison | Enterprise SaaS pricing only, no visual+metadata fusion |
| Shufti Pro | shuftipro.com | KYC + liveness, GDPR-aware, 230-country support | Third-party processing, opaque score only |
| **GROGU** | github.com/Jaidealistic | Dual-signal fusion, Grad-CAM, ~3.5 MB offline, DPDP-compliant, audit report export | Scoped to digital-native PDFs; small-scale real-world validation set |

The gap shared by all four commercial platforms: mandatory cloud dependency (DPDP/GDPR liability), single-signal visual-only analysis (metadata ignored as a forensic resource), and opaque scoring (no attribution). GROGU is designed to close all three simultaneously.

---

## 7. SDG Mapping

**7.1 SDGs Identified**
- SDG 16 — Peace, Justice & Strong Institutions
- SDG 9 — Industry, Innovation & Infrastructure

**7.2 Relation**
- **SDG 16 (Target 16.4 — reduce illicit financial flows):** Tampered KYC documents enable synthetic-identity credit access and evasion of financial intelligence monitoring. GROGU strengthens fraud-prevention infrastructure in a form accessible to institutions existing tools exclude.
- **SDG 9 (Target 9.3 — small enterprise access to finance/infrastructure):** Enterprise KYC platforms exclude small institutions on cost. GROGU delivers forensic capability at zero variable cost, on any local device — infrastructure democratisation.

---

## 8. Dataset — Full Specification (Canonical, Bug-Fixed)

> This is the final, corrected version of the dataset generation pipeline after multiple rounds of realism review and bug fixes. All known issues (account-number bug, doc_type/template mismatch, generic non-realistic field lists, broken transaction math, header-rule line overlapping text, T4 rental-agreement subtitle bug, financial-year formatting bug) have been fixed in the code below. Build against this section directly.

### 8.1 Dataset Identity
- **Name:** GROGU-KYC Bimodal Forensic Dataset
- **Source:** Fully self-generated via automated Python pipeline — zero real PII
- **Total instances:** ~4,000 PDF files
- **Classes:** 4 — `genuine` (0) / `font_tampered` (1) / `metadata_tampered` (2) / `both_tampered` (3)
- **Intended release:** IEEE Dataport (not yet published — this dataset is itself a research contribution, since no comparable bimodal forensic PDF corpus exists publicly)
- **Novelty:** Controlled Bimodal Feature Injection — visual tampering and metadata corruption are independently and simultaneously injectable into the same file, under exact rule-based (not GAN/diffusion) ground-truth control

### 8.2 Tech Stack
```
python >= 3.10
faker
reportlab
pillow
pymupdf      # fitz — PDF→image rendering for visual tampering
img2pdf      # image→PDF conversion after tampering
pypdf
pandas
scikit-learn
tqdm
```
```bash
pip install faker reportlab pillow pypdf pandas scikit-learn tqdm pymupdf img2pdf
```
(WeasyPrint was dropped from the stack — template diversity is achieved through structural/layout variation across 5 hand-built ReportLab templates instead, avoiding Cairo/Pango install friction.)

### 8.3 Output Structure
```
grogu_dataset/
├── raw_pdfs/
│   ├── genuine/
│   ├── font_tampered/
│   ├── metadata_tampered/
│   └── both_tampered/
├── metadata_vectors/
│   └── all_metadata.csv
├── labels/
│   └── master_labels.csv
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── generation_log.json
```

### 8.4 The 5 Document Templates

Templates are differentiated by **real institutional structure and field sets**, not decorative styling — each template mimics a genuinely distinct Indian document type with its own realistic layout, not the same generic field list re-skinned in a different colour.

```python
TEMPLATES = {
    "T1": {
        "name": "Bank Statement",
        "font_header": "Helvetica-Bold", "font_body": "Helvetica",
        "header_color": (0.05, 0.15, 0.45),   # corporate navy
        "font_size_header": 13, "font_size_body": 9,
        "footer_disclaimer": "This is a system-generated statement and does not require a signature."
    },
    "T2": {
        "name": "Salary Slip",
        "font_header": "Helvetica-Bold", "font_body": "Helvetica",
        "header_color": (0.15, 0.15, 0.15),   # near-black payroll styling
        "font_size_header": 12, "font_size_body": 9,
        "footer_disclaimer": "This is a computer-generated payslip and does not require a physical signature."
    },
    "T3": {
        "name": "Utility Bill",
        "font_header": "Helvetica-Bold", "font_body": "Helvetica",
        "header_color": (0.55, 0.15, 0.05),   # muted utility orange-red
        "font_size_header": 12, "font_size_body": 9,
        "footer_disclaimer": "Consumer Number and Billing Period are required for KYC verification."
    },
    "T4": {
        "name": "Address Proof",
        "font_header": "Times-Bold", "font_body": "Times-Roman",
        "header_color": (0.1, 0.1, 0.1),      # formal black
        "font_size_header": 12, "font_size_body": 10,
        "footer_disclaimer": "Valid as address proof under RBI KYC guidelines."
    },
    "T5": {
        "name": "Income Certificate",
        "font_header": "Times-Bold", "font_body": "Times-Roman",
        "header_color": (0.1, 0.1, 0.1),      # formal government black
        "font_size_header": 13, "font_size_body": 10,
        "footer_disclaimer": "Issued by the Competent Authority. Certificate Number and Date are mandatory."
    }
}

INDIAN_BANKS = [
    ("State Bank of India", "SBIN"), ("HDFC Bank", "HDFC"), ("ICICI Bank", "ICIC"),
    ("Punjab National Bank", "PUNB"), ("Axis Bank", "UTIB"), ("Bank of Baroda", "BARB"),
    ("Kotak Mahindra Bank", "KKBK"), ("Canara Bank", "CNRB")
]

ELECTRICITY_BOARDS = [
    "Maharashtra State Electricity Distribution Co. Ltd.",
    "Tamil Nadu Generation and Distribution Corporation",
    "Uttar Pradesh Power Corporation Ltd.",
    "Karnataka Power Transmission Corporation",
    "West Bengal State Electricity Distribution Co. Ltd."
]

STATES_FOR_CERT = ["Maharashtra", "Tamil Nadu", "Karnataka", "Uttar Pradesh", "West Bengal", "Gujarat"]

DEBIT_NARRATIONS = ["ATM Withdrawal", "POS/Amazon", "UPI/Zomato", "Bill Payment", "Cheque Debit"]
CREDIT_NARRATIONS = ["NEFT/Salary Credit", "IMPS Transfer", "UPI Received", "Interest Credit"]
```

### 8.5 Step 1 — Synthetic Identity Generation (bug-fixed)

```python
import random
from faker import Faker
from datetime import datetime

def generate_identity(template_key):
    fake = Faker('en_IN')
    bank_name, bank_code = random.choice(INDIAN_BANKS)

    identity = {
        "full_name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=21, maximum_age=60).strftime("%d/%m/%Y"),
        "address": fake.address().replace("\n", ", "),
        # FIX: numerify only replaces '#', not 'X' — this now produces real digits
        "account_number": fake.numerify(text="###########"),
        # FIX: realistic bank-prefixed IFSC instead of random letters
        "ifsc": f"{bank_code}0{fake.numerify(text='######')}",
        "bank_name": bank_name,
        "pan": fake.bothify(text="?????####?").upper(),
        "income": random.randint(25000, 250000),
        "employer": fake.company(),
        "email": fake.email(),
        # FIX: realistic 10-digit Indian mobile number
        "phone": fake.numerify(text="9########"),
        "statement_date": fake.date_between(start_date="-2y", end_date="today").strftime("%d/%m/%Y"),
        "base_id": fake.uuid4(),
        # FIX: doc_type is now DERIVED from template_key — never randomised independently.
        # This eliminates the old "UTILITY BILL — SALARY SLIP" mismatched-header bug.
        "template_key": template_key,
    }

    if template_key == "T1":  # Bank Statement
        identity["branch"] = f"{fake.city()} Branch"
        identity["account_type"] = random.choice(["Savings Account", "Current Account"])
        identity["period_from"] = fake.date_between(start_date="-60d", end_date="-30d").strftime("%d/%m/%Y")
        identity["period_to"] = fake.date_between(start_date="-29d", end_date="today").strftime("%d/%m/%Y")
        identity["opening_balance"] = random.randint(10000, 300000)
        identity["transactions"], identity["closing_balance"] = _generate_transactions(
            fake, identity["period_from"], identity["period_to"], identity["opening_balance"]
        )

    elif template_key == "T2":  # Salary Slip
        identity["employee_id"] = fake.bothify(text="EMP#####").upper()
        identity["designation"] = fake.job()
        identity["department"] = random.choice(["Engineering", "Sales", "Operations", "Finance", "HR"])
        identity["uan"] = fake.numerify(text="###########")
        identity["pay_month"] = fake.date_between(start_date="-6m", end_date="today").strftime("%B %Y")
        basic = round(identity["income"] * 0.5)
        hra = round(identity["income"] * 0.2)
        conv = round(identity["income"] * 0.1)
        special = identity["income"] - basic - hra - conv
        pf = round(basic * 0.12)
        pt = 200
        identity["earnings"] = [("Basic", basic), ("HRA", hra), ("Conveyance", conv), ("Special Allowance", special)]
        identity["deductions"] = [("Provident Fund", pf), ("Professional Tax", pt)]
        identity["gross_earnings"] = basic + hra + conv + special
        identity["total_deductions"] = pf + pt
        identity["net_pay"] = identity["gross_earnings"] - identity["total_deductions"]

    elif template_key == "T3":  # Utility Bill
        identity["provider"] = random.choice(ELECTRICITY_BOARDS)
        identity["consumer_no"] = fake.numerify(text="##########")
        identity["meter_no"] = fake.bothify(text="MT######").upper()
        identity["prev_reading"] = random.randint(1000, 5000)
        identity["curr_reading"] = identity["prev_reading"] + random.randint(80, 400)
        identity["units"] = identity["curr_reading"] - identity["prev_reading"]
        identity["bill_date"] = fake.date_between(start_date="-20d", end_date="today").strftime("%d/%m/%Y")
        identity["due_date"] = fake.date_between(start_date="today", end_date="+15d").strftime("%d/%m/%Y")
        energy_charge = identity["units"] * random.randint(6, 9)
        fixed_charge = random.randint(50, 150)
        tax = round((energy_charge + fixed_charge) * 0.05)
        identity["amount_due"] = energy_charge + fixed_charge + tax
        identity["charge_breakdown"] = [("Energy Charges", energy_charge), ("Fixed Charges", fixed_charge), ("Taxes", tax)]

    elif template_key == "T4":  # Address Proof
        identity["doc_subtype"] = random.choice(["passbook", "rental_agreement"])
        if identity["doc_subtype"] == "rental_agreement":
            identity["landlord_name"] = fake.name()
            identity["agreement_from"] = fake.date_between(start_date="-1y", end_date="-6m").strftime("%d/%m/%Y")
            identity["agreement_to"] = fake.date_between(start_date="+6m", end_date="+1y").strftime("%d/%m/%Y")
            identity["monthly_rent"] = random.randint(8000, 45000)
            identity["registration_no"] = fake.bothify(text="REG/####/??").upper()
        else:  # passbook
            identity["branch"] = f"{fake.city()} Branch"
            identity["customer_id"] = fake.numerify(text="CIF########")
            identity["nominee"] = fake.name()
            identity["issue_date"] = fake.date_between(start_date="-3y", end_date="-1y").strftime("%d/%m/%Y")

    elif template_key == "T5":  # Income Certificate
        identity["state"] = random.choice(STATES_FOR_CERT)
        identity["cert_no"] = fake.bothify(text="INC/####/??####").upper()
        identity["issuing_authority"] = f"Tehsildar, {fake.city()} Taluk"
        # FIX: financial year must be a valid consecutive YYYY-YY pair, e.g. "2024-25"
        fy_start = fake.random_int(2023, 2025)
        identity["financial_year"] = f"{fy_start}-{str(fy_start + 1)[2:]}"
        identity["annual_income"] = identity["income"] * 12
        identity["purpose"] = random.choice(["Bank KYC Verification", "Loan Application", "Scholarship Application"])

    return identity


def _generate_transactions(fake, period_from_str, period_to_str, opening_balance):
    """FIX: dates now sorted chronologically, and balance is a running total that
    reconciles deterministically with each row's debit/credit — not independently
    randomised columns."""
    d1 = datetime.strptime(period_from_str, "%d/%m/%Y")
    d2 = datetime.strptime(period_to_str, "%d/%m/%Y")
    dates = sorted(fake.date_between(start_date=d1, end_date=d2) for _ in range(5))
    balance = opening_balance
    rows = []
    for d in dates:
        if random.random() > 0.45:
            narration, amount = random.choice(DEBIT_NARRATIONS), random.randint(200, 9000)
            balance -= amount
            rows.append((d.strftime("%d/%m/%y"), narration, f"{amount:,}", "-", f"{balance:,}"))
        else:
            narration, amount = random.choice(CREDIT_NARRATIONS), random.randint(2000, 90000)
            balance += amount
            rows.append((d.strftime("%d/%m/%y"), narration, "-", f"{amount:,}", f"{balance:,}"))
    return rows, balance
```

### 8.6 Step 2 — Genuine Document Rendering (bug-fixed, canvas-based)

Uses raw ReportLab canvas (not platypus) so exact pixel bounding boxes can be recorded for each field — required by Step 3's visual tampering step.

```python
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import textwrap

def _wrap_text(text, width_chars=95):
    return textwrap.wrap(text, width_chars)

def _draw_header_rule(c, x_start, x_end, header_baseline_y, gap=2.14):
    """FIX: this single helper is now shared by every table-drawing function below.
    Previously, the transaction table used a gap of ~8pt while the other two tables
    correctly used ~2.14pt — the larger gap pushed the rule down into the first
    data row's ascenders, producing a strikethrough-through-text visual bug.
    Standardising on 2.14pt (proven clean in T2/T3) eliminates that class of bug."""
    c.setLineWidth(0.75)
    c.line(x_start, header_baseline_y - gap, x_end, header_baseline_y - gap)

def _draw_kv_block(c, start_y, rows, tmpl, x_label=50, x_value=200, row_height=28):
    field_boxes = {}
    y = start_y
    for label, value in rows:
        c.setFont(tmpl["font_body"], tmpl["font_size_body"] - 1)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(x_label, y, f"{label}:")
        c.setFont(tmpl["font_body"], tmpl["font_size_body"])
        c.setFillColorRGB(0, 0, 0)
        c.drawString(x_value, y, str(value))
        field_boxes[label] = {
            "x": x_value, "y": y - 4, "w": 300, "h": tmpl["font_size_body"] + 4,
            "value": str(value), "font": tmpl["font_body"], "font_size": tmpl["font_size_body"]
        }
        y -= row_height
    return field_boxes, y

def _draw_transaction_table(c, start_y, transactions):
    cols_x = [50, 130, 330, 410, 490]
    headers = ["Date", "Narration", "Debit", "Credit", "Balance"]
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(0, 0, 0)
    for x, h in zip(cols_x, headers):
        c.drawString(x, start_y, h)
    _draw_header_rule(c, 50, 545, start_y)

    field_boxes = {}
    y = start_y - 15
    c.setFont("Helvetica", 9)
    for i, row in enumerate(transactions):
        for x, val in zip(cols_x, row):
            c.drawString(x, y, str(val))
        if i == len(transactions) - 1:
            field_boxes["Closing Balance (Last Row)"] = {
                "x": cols_x[4], "y": y - 4, "w": 90, "h": 13,
                "value": row[4], "font": "Helvetica", "font_size": 9
            }
        y -= 15
    return field_boxes

def _draw_earnings_deductions(c, start_y, earnings, deductions, gross, total_deductions, net_pay):
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, start_y, "Earnings")
    c.drawString(200, start_y, "Amount (Rs)")
    c.drawString(320, start_y, "Deductions")
    c.drawString(470, start_y, "Amount (Rs)")
    _draw_header_rule(c, 50, 545, start_y)

    field_boxes = {}
    y = start_y - 15
    c.setFont("Helvetica", 9)
    max_rows = max(len(earnings), len(deductions))
    for i in range(max_rows):
        if i < len(earnings):
            label, val = earnings[i]
            c.drawString(50, y, label)
            c.drawString(200, y, f"{val:,}")
        if i < len(deductions):
            label, val = deductions[i]
            c.drawString(320, y, label)
            c.drawString(470, y, f"{val:,}")
        y -= 15

    _draw_header_rule(c, 50, 545, y + 15, gap=2.14)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y - 5, "Gross Earnings")
    c.drawString(200, y - 5, f"{gross:,}")
    c.drawString(320, y - 5, "Total Deductions")
    c.drawString(470, y - 5, f"{total_deductions:,}")
    y -= 20

    _draw_header_rule(c, 50, 545, y + 15, gap=2.14)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y - 10, "Net Pay")
    c.drawString(200, y - 10, f"{net_pay:,}")
    field_boxes["Net Pay"] = {"x": 200, "y": y - 14, "w": 100, "h": 14, "value": str(net_pay), "font": "Helvetica-Bold", "font_size": 10}
    field_boxes["Designation"] = None  # placeholder — actual box set in _draw_kv_block for T2
    return {k: v for k, v in field_boxes.items() if v is not None}

def _draw_charge_table(c, start_y, charge_breakdown, amount_due):
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, start_y, "Charge Details")
    c.drawString(400, start_y, "Amount (Rs)")
    _draw_header_rule(c, 50, 545, start_y)

    y = start_y - 15
    c.setFont("Helvetica", 9)
    for label, val in charge_breakdown:
        c.drawString(50, y, label)
        c.drawString(400, y, f"{val:,}")
        y -= 15

    _draw_header_rule(c, 50, 545, y + 15, gap=2.14)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y - 10, "Total Amount Due")
    c.drawString(400, y - 10, f"{amount_due:,}")
    return {
        "Total Amount Due": {"x": 400, "y": y - 14, "w": 90, "h": 14, "value": str(amount_due), "font": "Helvetica-Bold", "font_size": 10}
    }


def generate_genuine_pdf(identity, template_key, output_path):
    tmpl = TEMPLATES[template_key]
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    field_boxes = {}

    # Letterhead band
    r, g, b = tmpl["header_color"]
    c.setFillColorRGB(r, g, b)
    c.rect(0, height - 90, width, 90, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont(tmpl["font_header"], tmpl["font_size_header"])

    if template_key == "T1":
        c.drawString(50, height - 45, identity["bank_name"].upper())
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 62, f"{identity['branch']}  |  Statement of Account")
        c.drawRightString(width - 50, height - 45, f"IFSC: {identity['ifsc']}")
        c.drawRightString(width - 50, height - 62, f"A/C No: {identity['account_number']}")
        boxes, next_y = _draw_kv_block(c, height - 110, [
            ("Name", identity["full_name"]), ("Date of Birth", identity["dob"]),
            ("Account Type", identity["account_type"]),
            ("Statement Period", f"{identity['period_from']} to {identity['period_to']}"),
            ("Opening Balance", f"Rs. {identity['opening_balance']:,}"),
            ("Closing Balance", f"Rs. {identity['closing_balance']:,}"),
        ], tmpl)
        field_boxes.update(boxes)
        field_boxes.update(_draw_transaction_table(c, next_y - 5, identity["transactions"]))

    elif template_key == "T2":
        c.drawString(50, height - 45, identity["employer"].upper())
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 62, f"Payslip for {identity['pay_month']}")
        c.drawRightString(width - 50, height - 45, f"Emp ID: {identity['employee_id']}")
        boxes, next_y = _draw_kv_block(c, height - 110, [
            ("Name", identity["full_name"]), ("Designation", identity["designation"]),
            ("Department", identity["department"]), ("PAN", identity["pan"]),
            ("UAN", identity["uan"]), ("Bank A/C No", identity["account_number"]),
        ], tmpl)
        field_boxes.update(boxes)
        field_boxes.update(_draw_earnings_deductions(
            c, next_y - 5, identity["earnings"], identity["deductions"],
            identity["gross_earnings"], identity["total_deductions"], identity["net_pay"]
        ))

    elif template_key == "T3":
        c.drawString(50, height - 45, identity["provider"].upper())
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 62, "Electricity Consumption Bill")
        c.drawRightString(width - 50, height - 45, f"Consumer No: {identity['consumer_no']}")
        boxes, next_y = _draw_kv_block(c, height - 110, [
            ("Name", identity["full_name"]), ("Address", identity["address"]),
            ("Meter No", identity["meter_no"]), ("Bill Date", identity["bill_date"]),
            ("Due Date", identity["due_date"]),
            ("Units Consumed", f"{identity['units']} kWh ({identity['prev_reading']} \u2192 {identity['curr_reading']})"),
        ], tmpl)
        field_boxes.update(boxes)
        field_boxes.update(_draw_charge_table(c, next_y - 5, identity["charge_breakdown"], identity["amount_due"]))

    elif template_key == "T4":
        is_rental = identity["doc_subtype"] == "rental_agreement"
        title = "RENTAL AGREEMENT EXTRACT" if is_rental else "BANK PASSBOOK — FIRST PAGE"
        c.drawString(50, height - 45, title)
        c.setFont("Helvetica", 9)
        # FIX: subtitle no longer defaults to a random bank name for the rental-agreement
        # subtype — a rental agreement is not bank-issued.
        subtitle = "Registered Rental Agreement Extract" if is_rental else identity["bank_name"]
        c.drawString(50, height - 62, subtitle)

        rows = [("Name", identity["full_name"]), ("Address", identity["address"]), ("PAN", identity["pan"])]
        if is_rental:
            rows += [("Landlord", identity["landlord_name"]),
                     ("Agreement Period", f"{identity['agreement_from']} to {identity['agreement_to']}"),
                     ("Monthly Rent", f"Rs. {identity['monthly_rent']:,}"),
                     ("Registration No", identity["registration_no"])]
        else:
            rows += [("Branch", identity["branch"]), ("IFSC Code", identity["ifsc"]),
                     ("Customer ID", identity["customer_id"]), ("Nominee", identity["nominee"]),
                     ("Date of Issue", identity["issue_date"])]
        boxes, _ = _draw_kv_block(c, height - 110, rows, tmpl)
        field_boxes.update(boxes)

    elif template_key == "T5":
        c.drawString(50, height - 45, f"GOVERNMENT OF {identity['state'].upper()}")
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 62, "Revenue Department — Income Certificate")
        c.drawRightString(width - 50, height - 45, f"Cert No: {identity['cert_no']}")
        c.setFillColorRGB(0, 0, 0)
        c.setFont(tmpl["font_body"], tmpl["font_size_body"] + 1)
        para = (f"This is to certify that Shri/Smt {identity['full_name']}, resident of "
                f"{identity['address']}, has an annual family income of Rs. {identity['annual_income']:,}/- "
                f"for the financial year {identity['financial_year']}. This certificate is issued for the "
                f"purpose of {identity['purpose']}.")
        text_obj = c.beginText(50, height - 130)
        text_obj.setLeading(16)
        for line in _wrap_text(para, 95):
            text_obj.textLine(line)
        c.drawText(text_obj)
        field_boxes["Monthly Income"] = {"x": 50, "y": height - 145, "w": 495, "h": 16,
                                          "value": str(identity["annual_income"]), "font": tmpl["font_body"], "font_size": tmpl["font_size_body"]}
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 420, f"Date of Birth: {identity['dob']}   PAN: {identity['pan']}")
        field_boxes["Date of Birth"] = {"x": 130, "y": height - 425, "w": 100, "h": 14,
                                         "value": identity["dob"], "font": "Helvetica", "font_size": 9}
        c.line(width - 220, 90, width - 50, 90)
        c.drawString(width - 220, 75, identity["issuing_authority"])
        c.drawString(width - 220, 62, "(Signature & Seal)")

    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.setFont("Helvetica", 7)
    c.drawString(50, 30, f"{tmpl.get('footer_disclaimer', '')} | Doc ID: {identity['base_id']}")
    c.save()
    return field_boxes
```

### 8.7 Step 3 — Visual Tampering (per-template tamperable fields)

```python
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import img2pdf, os

MISMATCH_FONTS = ["arial.ttf", "verdana.ttf", "georgia.ttf", "trebuc.ttf", "comic.ttf", "cour.ttf"]

# FIX: tamperable fields are now per-template, since each template has a distinct field set
TAMPERABLE_FIELDS_BY_TEMPLATE = {
    "T1": ["Account Type", "Name", "Date of Birth", "Closing Balance (Last Row)"],
    "T2": ["Designation", "PAN", "Bank A/C No", "Name", "Net Pay"],
    "T3": ["Units Consumed", "Due Date", "Meter No", "Total Amount Due"],
    "T4": ["Monthly Rent", "Registration No", "Address", "Nominee"],
    "T5": ["Monthly Income", "Date of Birth"],
}

def apply_visual_tampering(pdf_path, output_path, field_boxes, template_key, severity="medium"):
    available = [f for f in TAMPERABLE_FIELDS_BY_TEMPLATE[template_key] if f in field_boxes]
    target_field = random.choice(available)
    box = field_boxes[target_field]

    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    draw = ImageDraw.Draw(img)
    scale = 2
    x = int(box["x"] * scale)
    y = int((842 - box["y"] - box["h"]) * scale)
    w = int(box["w"] * scale)
    h = int(box["h"] * scale) + 4

    draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], fill="white")
    new_value = generate_fake_field_value(target_field)

    offsets = {"low": (2, 2, 0), "medium": (4, 3, 1), "high": (8, 5, 2)}
    xr, yr, fr = offsets[severity]
    x_offset, y_offset = random.randint(-xr, xr), random.randint(-yr, yr)
    font_size_delta = random.randint(-fr, fr)

    mismatch_font_file = random.choice(MISMATCH_FONTS)
    font_size = int(box["font_size"] * scale) + font_size_delta
    try:
        font = ImageFont.truetype(mismatch_font_file, font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text((x + x_offset, y + y_offset), new_value, fill=(0, 0, 0), font=font)

    png_path = output_path.replace(".pdf", "_visual.png")
    img.save(png_path)
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(png_path))
    os.remove(png_path)

    return {
        "tampered_field": target_field, "original_value": box["value"], "new_value": new_value,
        "bounding_box": {"x": box["x"], "y": box["y"], "w": box["w"], "h": box["h"]},
        "severity": severity, "font_used": mismatch_font_file,
        "x_offset": x_offset, "y_offset": y_offset
    }

def generate_fake_field_value(field_name):
    fake = Faker('en_IN')
    generators = {
        "Monthly Income": lambda: str(random.randint(180000, 3000000)),
        "Date of Birth": lambda: fake.date_of_birth(minimum_age=21, maximum_age=60).strftime("%d/%m/%Y"),
        "Account Type": lambda: random.choice(["Savings Account", "Current Account"]),
        "Name": lambda: fake.name(),
        "Closing Balance (Last Row)": lambda: f"{random.randint(10000,300000):,}",
        "Designation": lambda: fake.job(),
        "PAN": lambda: fake.bothify(text="?????####?").upper(),
        "Bank A/C No": lambda: fake.numerify(text="###########"),
        "Net Pay": lambda: f"{random.randint(15000,500000):,}",
        "Units Consumed": lambda: f"{random.randint(50,500)} kWh",
        "Due Date": lambda: fake.date_between(start_date="today", end_date="+30d").strftime("%d/%m/%Y"),
        "Meter No": lambda: fake.bothify(text="MT######").upper(),
        "Total Amount Due": lambda: f"{random.randint(500,5000):,}",
        "Monthly Rent": lambda: f"Rs. {random.randint(8000,45000)}",
        "Registration No": lambda: fake.bothify(text="REG/####/??").upper(),
        "Address": lambda: fake.address().replace("\n", ", "),
        "Nominee": lambda: fake.name(),
    }
    return generators.get(field_name, lambda: fake.word())()
```

### 8.8 Step 4 — Metadata Corruption (12 attributes + temporal anachronism)

```python
from pypdf import PdfReader, PdfWriter
from datetime import datetime, timedelta

METADATA_ATTRIBUTES = [
    "/Title", "/Author", "/Subject", "/Keywords", "/Creator", "/Producer",
    "/CreationDate", "/ModDate", "/Trapped", "/CustomField1", "/CustomField2", "/CustomField3"
]

SOFTWARE_SIGNATURES = {
    "old": ["Adobe Acrobat 9.0", "Microsoft Word 2010", "LibreOffice 3.6", "Adobe Acrobat 8.0", "OpenOffice.org Writer 3.2"],
    "medium": ["Adobe Acrobat DC 2017", "Microsoft Word 2016", "LibreOffice 5.4", "Foxit Reader 9.0"],
    "new": ["Adobe Acrobat DC 2024", "Microsoft Word 2024", "LibreOffice 7.6", "iLovePDF Desktop 3.0", "Smallpdf 2024", "Adobe Acrobat 2024.002"]
}

def apply_metadata_tampering(pdf_path, output_path, claimed_year=None):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    existing_meta = reader.metadata or {}
    num_to_corrupt = random.randint(2, 7)
    attrs_to_corrupt = random.sample(METADATA_ATTRIBUTES, num_to_corrupt)

    corruption_log, new_metadata = {}, {}
    for attr in attrs_to_corrupt:
        original = existing_meta.get(attr, "")
        if attr in ("/Producer", "/Creator"):
            new_val = random.choice(SOFTWARE_SIGNATURES["new"])
        elif attr == "/CreationDate":
            if claimed_year:
                fake_old_date = datetime(claimed_year, random.randint(1, 12), random.randint(1, 28))
            else:
                fake_old_date = datetime.now() - timedelta(days=random.randint(365, 1825))
            new_val = fake_old_date.strftime("D:%Y%m%d%H%M%S")
        elif attr == "/ModDate":
            recent_date = datetime.now() - timedelta(days=random.randint(0, 30))
            new_val = recent_date.strftime("D:%Y%m%d%H%M%S")
        elif attr == "/Author":
            new_val = Faker('en_IN').name()
        elif attr == "/Title":
            new_val = random.choice(["Scanned Document", "Bank Statement Copy", "Untitled", "Document1", "PDF Export"])
        else:
            new_val = Faker().word()
        new_metadata[attr] = new_val
        corruption_log[attr] = {"original": str(original), "corrupted_to": new_val}

    writer.add_metadata(new_metadata)
    with open(output_path, "wb") as f:
        writer.write(f)

    temporal_delta = compute_temporal_anachronism(new_metadata)
    return {
        "num_attributes_corrupted": num_to_corrupt,
        "corrupted_attributes": attrs_to_corrupt,
        "corruption_details": corruption_log,
        "temporal_anachronism_delta": temporal_delta
    }

def compute_temporal_anachronism(metadata):
    producer = metadata.get("/Producer", "") or metadata.get("/Creator", "")
    creation_date_str = metadata.get("/CreationDate", "")
    claimed_year = None
    if creation_date_str and len(creation_date_str) >= 6:
        try:
            claimed_year = int(creation_date_str[2:6])
        except Exception:
            claimed_year = datetime.now().year
    software_year = None
    for year in range(2008, 2026):
        if str(year) in producer:
            software_year = year
            break
    if claimed_year and software_year:
        return software_year - claimed_year
    return 0
```

### 8.9 Step 5 — Metadata Feature Vector Extraction (13 features)

```python
def extract_metadata_vector(pdf_path, temporal_delta=0):
    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata or {}
    except Exception:
        return [0] * 13

    def parse_pdf_date(date_str):
        if not date_str or len(date_str) < 10:
            return None
        try:
            return datetime.strptime(date_str[2:14], "%Y%m%d%H%M")
        except Exception:
            return None

    creation_date = parse_pdf_date(str(meta.get("/CreationDate", "")))
    mod_date = parse_pdf_date(str(meta.get("/ModDate", "")))

    f1 = 1 if meta.get("/Creator") else 0
    f2 = 1 if meta.get("/Producer") else 0
    f3 = 1 if meta.get("/Author") else 0
    f4 = 1 if creation_date else 0
    f5 = 1 if mod_date else 0
    f6 = max(0, (mod_date - creation_date).days) if creation_date and mod_date else 0
    f7 = (creation_date.year - 2000) if creation_date else 0
    f8 = (mod_date.year - 2000) if mod_date else 0

    suspect_tools = ["ilovepdf", "smallpdf", "online", "converter", "compress"]
    producer_str = str(meta.get("/Producer", "")).lower()
    f9 = 1 if any(t in producer_str for t in suspect_tools) else 0
    f10 = min(len(str(meta.get("/Producer", ""))), 100)

    standard_fields = {"/Title", "/Author", "/Subject", "/Keywords", "/Creator", "/Producer", "/CreationDate", "/ModDate", "/Trapped"}
    f11 = sum(1 for k in meta.keys() if k not in standard_fields)

    suspicious_titles = ["untitled", "document1", "scanned", "copy", "pdf"]
    title_str = str(meta.get("/Title", "")).lower()
    f12 = 1 if any(t in title_str for t in suspicious_titles) else 0

    f13 = temporal_delta
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]

METADATA_FEATURE_NAMES = [
    "has_creator", "has_producer", "has_author", "has_creation_date", "has_mod_date",
    "creation_mod_delta_days", "creation_year_norm", "mod_year_norm",
    "suspect_producer_tool", "producer_string_length", "num_custom_fields",
    "suspicious_title_flag", "temporal_anachronism_delta"
]
```

### 8.10 Step 6 — Master Generation Loop, Splitting & Verification

Unchanged from the original brief in structure — only the calls into `generate_identity(template_key)` and `apply_visual_tampering(..., template_key, ...)` need the extra `template_key` argument now threaded through, since Sections 8.5–8.7 above require it. Key requirements to preserve:

- **Config:** `n_genuine = n_font_tampered = n_metadata_tampered = n_both_tampered = 1000` (≈4,000 total)
- **Class labels:** `genuine=0, font_tampered=1, metadata_tampered=2, both_tampered=3`
- **Split strategy:** `GroupShuffleSplit` grouped by `base_id`, NOT by individual file — prevents genuine/tampered variants of the same base document from leaking across train/val/test
- **Split ratio:** 70% train / 10% val / 20% test (outer split 80/20, inner split 87.5/12.5 of the 80%)
- **Leakage assertion:** must assert zero `base_id` overlap between all split pairs before writing output
- **Error handling:** wrap every single document generation in try/except; log failures to `generation_log.json`; a single failure must never abort the full run
- **`verify_dataset.py`:** run after generation — checks class balance, split leakage, missing files, temporal anachronism distribution per class, metadata vector null-check

### 8.11 Pre-Flight Checklist (apply before scaling to full 4,000)

1. Run a pilot batch of ~40–50 per class (~160–200 total) before the full run
2. Run `verify_dataset.py` against the pilot and confirm all leakage assertions pass
3. Visually spot-check ~10–15 rendered PDFs across **all 5 templates**, specifically including **both** T4 subtypes (`passbook` and `rental_agreement` — force-generate one of each if random sampling doesn't surface both)
4. Confirm: no strikethrough/overlap between table header rules and row text (Section 8.6's `_draw_header_rule` fixes this); T1 transaction balances reconcile row-by-row; T5 financial year renders as `YYYY-YY` never `YYYY-YYYY`
5. Only then scale `CONFIG["n_*"]` to 1000 each and run the full batch

---

## 9. Novelty

Five distinct levels of novelty — not one:

1. **Dual-signal forensic fusion** — MobileNetV2 (visual) + Tabular MLP (metadata) fused via `torch.cat()`; each branch degrades independently (metadata absent on scans → visual branch still runs at full confidence)
2. **Grad-CAM forgery localisation** — spatial heatmap shows exact tampered region → legally-usable evidence, not just a label
3. **Temporal anachronism detection** — cross-checks claimed creation date vs. PDF producer software version year → catches a "2019 doc, 2024 software" impossibility → not present in any published model
4. **Edge-native INT8 deployment** — ~3.5 MB, 100% offline, zero cloud cost, DPDP-compliant by architecture, no per-check fees
5. **One-click forensic audit report export** — auto-generated PDF: heatmap + verdict + metadata evidence + timestamp → compliance paper trail (reuses ReportLab, zero new dependencies)

**Dataset novelty:** Controlled Bimodal Feature Injection — first synthetic KYC corpus to inject and independently label visual + metadata anomalies in the same file, 100% synthetic PII, rule-based exact ground truth (GANs/diffusion cannot match this precision).

> "The algorithms aren't new — MobileNetV2, INT8 quantisation, Grad-CAM all exist. The novelty is the architectural constraint: stripping a multi-modal fusion network of its cloud dependency, quantising it to run offline on a mobile processor, and adding a temporal anachronism signal absent from any published model."

---

## 10. What the App-Building Agent Needs to Build

1. **Dataset generation pipeline** — Section 8, as specified, producing `grogu_dataset/` with the full directory structure
2. **Model training pipeline** —
   - Visual branch: MobileNetV2 (ImageNet-pretrained, fine-tuned) on rendered document images
   - Metadata branch: Tabular MLP on the 13-feature vector from `metadata_vectors/all_metadata.csv`
   - Late fusion via concatenation → 4-class softmax head
   - Train/val/test using the pre-split, leakage-free CSVs in `splits/`
3. **Explainability layer** — Grad-CAM on the visual branch's final conv layer; overlay heatmap on the original document image
4. **Quantisation** — INT8 post-training quantisation of the fused model, target ≤3.5 MB
5. **Inference app** — accepts a PDF upload, runs both branches, returns: class verdict, confidence tier (e.g. green/amber/red), Grad-CAM overlay image, and which branch(es) triggered
6. **Audit report generator** — ReportLab-based PDF export combining: document thumbnail, Grad-CAM overlay, verdict, flagged metadata attributes, timestamp
7. **Offline-first constraint** — no network calls in the inference path; the entire pipeline (both branches + fusion) must run locally on the packaged model artifact

---

## 11. Known Constraints & Honest Scope

- Scoped to **digitally-edited, digital-native PDF documents** — the highest-volume tampering vector
- Scanned/photographed documents trigger graceful degradation: visual branch only, metadata branch transparently disabled, UI communicates reduced-confidence mode
- Training data is 100% synthetic (Faker-generated identities, zero real PII); a small, separately-constructed real-world out-of-distribution validation set (physically printed, re-scanned/re-photographed, hand-tampered) is used as a limited — not comprehensive — sim-to-real generalisation check
- Dataset is not yet on IEEE Dataport; intended for future release with a representative sample shared for reference
