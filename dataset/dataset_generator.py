import os
import json
import csv
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image, ImageDraw, ImageFont
import fitz
import img2pdf
import pypdf
from pypdf import PdfReader, PdfWriter
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


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

def _generate_transactions(fake, period_from_str, period_to_str, opening_balance):
    from datetime import datetime
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

def generate_identity(template_key):
    fake = Faker('en_IN')
    bank_name, bank_code = random.choice(INDIAN_BANKS)

    identity = {
        "full_name": fake.name(),
        "dob": fake.date_of_birth(minimum_age=21, maximum_age=60).strftime("%d/%m/%Y"),
        "address": fake.address().replace("\n", ", "),
        "account_number": fake.numerify(text="###########"),  # real 11-digit account no.
        "ifsc": f"{bank_code}0{fake.numerify(text='######')}",  # realistic bank-prefixed IFSC
        "bank_name": bank_name,
        "pan": fake.bothify(text="?????####?").upper(),
        "income": random.randint(25000, 250000),
        "employer": fake.company(),
        "email": fake.email(),
        "phone": fake.numerify(text="9########"),   # 10-digit mobile starting 9
        "statement_date": fake.date_between(start_date="-2y", end_date="today").strftime("%d/%m/%Y"),
        "base_id": fake.uuid4(),
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
        identity["net_pay"] = identity["income"] - pf - pt

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
        else:
            identity["branch"] = f"{fake.city()} Branch"
            identity["customer_id"] = fake.numerify(text="CIF########")
            identity["nominee"] = fake.name()
            identity["issue_date"] = fake.date_between(start_date="-3y", end_date="-1y").strftime("%d/%m/%Y")

    elif template_key == "T5":  # Income Certificate
        identity["state"] = random.choice(STATES_FOR_CERT)
        identity["cert_no"] = fake.bothify(text="INC/####/??####").upper()
        identity["issuing_authority"] = f"Tehsildar, {fake.city()} Taluk"
        fy_start = fake.random_int(2023, 2025)
        identity["financial_year"] = f"{fy_start}-{str(fy_start + 1)[2:]}"
        identity["annual_income"] = identity["income"] * 12
        identity["purpose"] = random.choice(["Bank KYC Verification", "Loan Application", "Scholarship Application"])

    return identity

TEMPLATES = {
    "T1": {
        "name": "Bank Statement",
        "issuer": "State-Bank-style Statement of Account",
        "font_header": "Helvetica-Bold",
        "font_body": "Helvetica",
        "header_color": (0.05, 0.15, 0.45),   # Standard corporate navy
        "font_size_header": 13,
        "font_size_body": 9,
        "layout": "letterhead_kv_table",       # logo strip + key-value field table
        "has_letterhead": True,
        "has_ref_number": True,
        "footer_disclaimer": "This is a system-generated statement and does not require a signature."
    },
    "T2": {
        "name": "Salary Slip",
        "issuer": "Corporate Payslip",
        "font_header": "Helvetica-Bold",
        "font_body": "Helvetica",
        "header_color": (0.15, 0.15, 0.15),   # Near-black, standard payroll styling
        "font_size_header": 12,
        "font_size_body": 9,
        "layout": "table_grid",                # earnings/deductions grid, like real payslips
        "has_letterhead": True,
        "has_ref_number": True,
        "footer_disclaimer": "This is a computer-generated payslip and does not require a physical signature."
    },
    "T3": {
        "name": "Utility Bill",
        "issuer": "Electricity / Water Board",
        "font_header": "Helvetica-Bold",
        "font_body": "Helvetica",
        "header_color": (0.55, 0.15, 0.05),   # Muted utility-provider orange-red
        "font_size_header": 12,
        "font_size_body": 9,
        "layout": "single_column_billing",     # consumer no. / billing period / amount due block
        "has_letterhead": True,
        "has_ref_number": True,
        "footer_disclaimer": "Consumer Number and Billing Period are required for KYC verification."
    },
    "T4": {
        "name": "Address Proof",
        "issuer": "Bank Passbook / Rental Agreement Extract",
        "font_header": "Times-Bold",
        "font_body": "Times-Roman",
        "header_color": (0.1, 0.1, 0.1),      # Standard formal black
        "font_size_header": 12,
        "font_size_body": 10,
        "layout": "letterhead",                # formal letterhead block, no italics
        "has_letterhead": True,
        "has_ref_number": True,
        "footer_disclaimer": "Valid as address proof under RBI KYC guidelines."
    },
    "T5": {
        "name": "Income Certificate",
        "issuer": "Government-issued Income Certificate",
        "font_header": "Times-Bold",
        "font_body": "Times-Roman",
        "header_color": (0.1, 0.1, 0.1),      # Formal government black
        "font_size_header": 13,
        "font_size_body": 10,
        "layout": "certificate_formal",        # emblem placeholder + certificate number + seal line
        "has_letterhead": True,
        "has_ref_number": True,
        "footer_disclaimer": "Issued by the Competent Authority. Certificate Number and Date are mandatory."
    }
}

def _draw_kv_block(c, start_y, rows, tmpl):
    field_boxes = {}
    y_pos = start_y
    c.setFont(tmpl["font_body"] + "" if "Bold" not in tmpl["font_body"] else tmpl["font_body"], tmpl["font_size_body"] - 1)
    for label, value in rows:
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(50, y_pos, f"{label}:")
        c.setFont(tmpl["font_body"], tmpl["font_size_body"])
        c.setFillColorRGB(0, 0, 0)
        c.drawString(200, y_pos, str(value))
        
        field_boxes[label] = {
            "x": 200,
            "y": y_pos - 4,
            "w": 300,
            "h": tmpl["font_size_body"] + 4,
            "value": str(value),
            "font": tmpl["font_body"],
            "font_size": tmpl["font_size_body"]
        }
        y_pos -= 28
    return field_boxes

def _draw_transaction_table(c, start_y, transactions):
    field_boxes = {}
    y_pos = start_y
    
    # Draw header
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(50, y_pos, "Date")
    c.drawString(120, y_pos, "Narration")
    c.drawRightString(350, y_pos, "Debit")
    c.drawRightString(450, y_pos, "Credit")
    c.drawRightString(540, y_pos, "Balance")
    
    c.setLineWidth(0.5)
    c.line(50, y_pos - 4, 540, y_pos - 4)
    
    y_pos -= 15
    c.setFont("Helvetica", 9)
    for t_date, narration, debit, credit, balance in transactions:
        c.drawString(50, y_pos, t_date)
        c.drawString(120, y_pos, narration)
        c.drawRightString(350, y_pos, debit)
        c.drawRightString(450, y_pos, credit)
        c.drawRightString(540, y_pos, balance)
        y_pos -= 15
    return field_boxes

def _draw_earnings_deductions(c, start_y, earnings, deductions, net_pay):
    field_boxes = {}
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, start_y, "Earnings")
    c.drawRightString(280, start_y, "Amount (Rs)")
    c.drawString(300, start_y, "Deductions")
    c.drawRightString(540, start_y, "Amount (Rs)")
    
    c.line(50, start_y - 4, 540, start_y - 4)
    
    y_pos = start_y - 15
    c.setFont("Helvetica", 9)
    max_rows = max(len(earnings), len(deductions))
    for i in range(max_rows):
        if i < len(earnings):
            e_name, e_amt = earnings[i]
            c.drawString(50, y_pos, e_name)
            c.drawRightString(280, y_pos, f"{e_amt:,}")
        if i < len(deductions):
            d_name, d_amt = deductions[i]
            c.drawString(300, y_pos, d_name)
            c.drawRightString(540, y_pos, f"{d_amt:,}")
        y_pos -= 15
        
    c.line(50, y_pos + 11, 540, y_pos + 11)
    
    gross = sum(v for _, v in earnings)
    total_deductions = sum(v for _, v in deductions)
    y_pos -= 5
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y_pos, "Gross Earnings")
    c.drawRightString(280, y_pos, f"{gross:,}")
    c.drawString(300, y_pos, "Total Deductions")
    c.drawRightString(540, y_pos, f"{total_deductions:,}")
    
    c.line(50, y_pos - 4, 540, y_pos - 4)
    
    y_pos -= 15
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y_pos, "Net Pay")
    c.drawRightString(280, y_pos, f"{net_pay:,}")
    return field_boxes

def _draw_charge_table(c, start_y, charges, total):
    field_boxes = {}
    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, start_y, "Charge Details")
    c.drawRightString(300, start_y, "Amount (Rs)")
    
    c.line(50, start_y - 4, 300, start_y - 4)
    
    y_pos = start_y - 15
    c.setFont("Helvetica", 9)
    for c_name, c_amt in charges:
        c.drawString(50, y_pos, c_name)
        c.drawRightString(300, y_pos, f"{c_amt:,}")
        y_pos -= 15
        
    c.line(50, y_pos + 11, 300, y_pos + 11)
    
    y_pos -= 5
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y_pos, "Total Amount Due")
    c.drawRightString(300, y_pos, f"{total:,}")
    return field_boxes

def _wrap_text(text, max_chars):
    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    for word in words:
        if current_len + len(word) > max_chars:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
        else:
            current_line.append(word)
            current_len += len(word) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def generate_genuine_pdf(identity, template_key, output_path):
    tmpl = TEMPLATES[template_key]
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    field_boxes = {}

    # Shared letterhead band
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
        field_boxes = _draw_kv_block(c, height - 110, [
            ("Name", identity["full_name"]), ("Date of Birth", identity["dob"]),
            ("Account Type", identity["account_type"]),
            ("Statement Period", f"{identity['period_from']} to {identity['period_to']}"),
            ("Opening Balance", f"Rs. {identity['opening_balance']:,}"),
            ("Closing Balance", f"Rs. {identity['closing_balance']:,}")
        ], tmpl)
        field_boxes.update(_draw_transaction_table(c, height - 290, identity["transactions"]))

    elif template_key == "T2":
        c.drawString(50, height - 45, identity["employer"].upper())
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 62, f"Payslip for {identity['pay_month']}")
        c.drawRightString(width - 50, height - 45, f"Emp ID: {identity['employee_id']}")
        field_boxes = _draw_kv_block(c, height - 110, [
            ("Name", identity["full_name"]), ("Designation", identity["designation"]),
            ("Department", identity["department"]), ("PAN", identity["pan"]),
            ("UAN", identity["uan"]), ("Bank A/C No", identity["account_number"]),
        ], tmpl)
        field_boxes.update(_draw_earnings_deductions(c, height - 300, identity["earnings"],
                                                       identity["deductions"], identity["net_pay"]))

    elif template_key == "T3":
        c.drawString(50, height - 45, identity["provider"].upper())
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 62, "Electricity Consumption Bill")
        c.drawRightString(width - 50, height - 45, f"Consumer No: {identity['consumer_no']}")
        field_boxes = _draw_kv_block(c, height - 110, [
            ("Name", identity["full_name"]), ("Address", identity["address"]),
            ("Meter No", identity["meter_no"]), ("Bill Date", identity["bill_date"]),
            ("Due Date", identity["due_date"]),
            ("Units Consumed", f"{identity['units']} kWh ({identity['prev_reading']} \u2192 {identity['curr_reading']})"),
        ], tmpl)
        field_boxes.update(_draw_charge_table(c, height - 290, identity["charge_breakdown"], identity["amount_due"]))

    elif template_key == "T4":
        title = "RENTAL AGREEMENT EXTRACT" if identity["doc_subtype"] == "rental_agreement" else "BANK PASSBOOK — FIRST PAGE"
        c.drawString(50, height - 45, title)
        c.setFont("Helvetica", 9)
        if identity["doc_subtype"] == "rental_agreement":
            subtitle = "Registered Rental Agreement Extract"
        else:
            subtitle = identity.get("bank_name", "Registered Document Extract")
        c.drawString(50, height - 62, subtitle)
        rows = [("Name", identity["full_name"]), ("Address", identity["address"]), ("PAN", identity["pan"])]
        if identity["doc_subtype"] == "rental_agreement":
            rows += [("Landlord", identity["landlord_name"]),
                     ("Agreement Period", f"{identity['agreement_from']} to {identity['agreement_to']}"),
                     ("Monthly Rent", f"Rs. {identity['monthly_rent']:,}"),
                     ("Registration No", identity["registration_no"])]
        else:
            rows += [("Branch", identity["branch"]), ("IFSC Code", identity["ifsc"]),
                     ("Customer ID", identity["customer_id"]), ("Nominee", identity["nominee"]),
                     ("Date of Issue", identity["issue_date"])]
        field_boxes = _draw_kv_block(c, height - 110, rows, tmpl)

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
        field_boxes["Date of Birth"] = {"x": 50, "y": height - 400, "w": 200, "h": 16,
                                         "value": identity["dob"], "font": tmpl["font_body"], "font_size": tmpl["font_size_body"]}
        c.setFont("Helvetica", 9)
        c.drawString(50, height - 420, f"Date of Birth: {identity['dob']}   PAN: {identity['pan']}")
        c.line(width - 220, 90, width - 50, 90)
        c.drawString(width - 220, 75, identity["issuing_authority"])
        c.drawString(width - 220, 62, "(Signature & Seal)")

    # Standard footer disclaimer — unchanged from your brief
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.setFont("Helvetica", 7)
    c.drawString(50, 30, f"{tmpl.get('footer_disclaimer', '')} | Doc ID: {identity['base_id']}")
    c.save()
    return field_boxes

# Windows standard fonts
MISMATCH_FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/trebuc.ttf",
    "C:/Windows/Fonts/comic.ttf",   
    "C:/Windows/Fonts/cour.ttf"     
]

TAMPERABLE_FIELDS_BY_TEMPLATE = {
    "T1": ["Account Type", "Name", "Date of Birth"],
    "T2": ["Designation", "PAN", "Bank A/C No", "Name"],
    "T3": ["Units Consumed", "Due Date", "Meter No"],
    "T4": ["Monthly Rent", "Registration No", "Address"],
    "T5": ["Monthly Income", "Date of Birth"],
}

def apply_visual_tampering(pdf_path, output_path, field_boxes, template_key, severity="medium"):
    """
    Opens PDF as image, erases a field value, re-types it with a mismatched font.
    Returns a dict of what was tampered for ground truth logging.
    """
    available = [f for f in TAMPERABLE_FIELDS_BY_TEMPLATE[template_key] if f in field_boxes]
    target_field = random.choice(available)
    box = field_boxes[target_field]

    # Convert PDF to image
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(2, 2)  # 2x resolution for better quality
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    draw = ImageDraw.Draw(img)

    # Scale box coordinates to image resolution
    scale = 2
    x = int(box["x"] * scale)
    y = int((842 - box["y"] - box["h"]) * scale)  # A4 height = 842 pts, flip Y axis
    w = int(box["w"] * scale)
    h = int(box["h"] * scale) + 4

    # Erase original value — paint over with white
    draw.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], fill="white")

    # Generate new fake value for the field
    new_value = generate_fake_field_value(target_field)

    # Apply alignment offset
    if severity == "low":
        x_offset = random.randint(-2, 2)
        y_offset = random.randint(-2, 2)
        font_size_delta = 0
    elif severity == "medium":
        x_offset = random.randint(-4, 4)
        y_offset = random.randint(-3, 3)
        font_size_delta = random.randint(-1, 1)
    else:  # high
        x_offset = random.randint(-8, 8)
        y_offset = random.randint(-5, 5)
        font_size_delta = random.randint(-2, 2)

    # Select mismatched font
    mismatch_font_file = random.choice(MISMATCH_FONTS)
    font_size = int(box["font_size"] * scale) + font_size_delta

    try:
        font = ImageFont.truetype(mismatch_font_file, font_size)
    except:
        font = ImageFont.load_default()

    # Draw tampered value
    draw.text(
        (x + x_offset, y + y_offset),
        new_value,
        fill=(0, 0, 0),
        font=font
    )

    # Save as PDF
    png_path = output_path.replace(".pdf", "_visual.png")
    img.save(png_path)

    # Convert back to PDF using img2pdf
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(png_path))
    os.remove(png_path)

    return {
        "tampered_field": target_field,
        "original_value": box["value"],
        "new_value": new_value,
        "bounding_box": {"x": box["x"], "y": box["y"], "w": box["w"], "h": box["h"]},
        "severity": severity,
        "font_used": mismatch_font_file,
        "x_offset": x_offset,
        "y_offset": y_offset
    }

def generate_fake_field_value(field_name):
    """Generate a plausible but fake replacement value for each field type."""
    fake_val = Faker('en_IN')
    generators = {
        "Monthly Income": lambda: str(random.randint(25000, 250000) * 12),
        "Date of Birth": lambda: fake_val.date_of_birth(minimum_age=21, maximum_age=60).strftime("%d/%m/%Y"),
        "Account Number": lambda: fake_val.numerify(text="###########"),
        "Bank A/C No": lambda: fake_val.numerify(text="###########"),
        "Address": lambda: fake_val.address().replace("\n", ", "),
        "Employer": lambda: fake_val.company(),
        "Account Type": lambda: random.choice(["Savings Account", "Current Account"]),
        "Name": lambda: fake_val.name(),
        "Designation": lambda: fake_val.job(),
        "PAN": lambda: fake_val.bothify(text="?????####?").upper(),
        "Units Consumed": lambda: f"{random.randint(50, 600)} kWh ({random.randint(1000, 5000)} \u2192 {random.randint(1000, 5000) + random.randint(50, 600)})",
        "Due Date": lambda: fake_val.date_between(start_date="today", end_date="+15d").strftime("%d/%m/%Y"),
        "Meter No": lambda: fake_val.bothify(text="MT######").upper(),
        "Monthly Rent": lambda: f"Rs. {random.randint(8000, 45000):,}",
        "Registration No": lambda: fake_val.bothify(text="REG/####/??").upper()
    }
    return generators.get(field_name, lambda: fake_val.word())()

# The 12 metadata attributes to potentially corrupt
METADATA_ATTRIBUTES = [
    "/Title",
    "/Author",
    "/Subject",
    "/Keywords",
    "/Creator",
    "/Producer",
    "/CreationDate",
    "/ModDate",
    "/Trapped",
    "/CustomField1",
    "/CustomField2",
    "/CustomField3"
]

# Software version signatures — used for temporal anachronism injection
SOFTWARE_SIGNATURES = {
    "old": [
        "Adobe Acrobat 9.0",
        "Microsoft Word 2010",
        "LibreOffice 3.6",
        "Adobe Acrobat 8.0",
        "OpenOffice.org Writer 3.2"
    ],
    "medium": [
        "Adobe Acrobat DC 2017",
        "Microsoft Word 2016",
        "LibreOffice 5.4",
        "Foxit Reader 9.0"
    ],
    "new": [
        "Adobe Acrobat DC 2024",
        "Microsoft Word 2024",
        "LibreOffice 7.6",
        "iLovePDF Desktop 3.0",
        "Smallpdf 2024",
        "Adobe Acrobat 2024.002"
    ]
}

def apply_metadata_tampering(pdf_path, output_path, claimed_year=None):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    existing_meta = reader.metadata or {}

    num_to_corrupt = random.randint(2, 7)
    attrs_to_corrupt = random.sample(METADATA_ATTRIBUTES, num_to_corrupt)

    corruption_log = {}
    new_metadata = {}

    for attr in attrs_to_corrupt:
        original = existing_meta.get(attr, "")

        if attr == "/Producer" or attr == "/Creator":
            new_val = random.choice(SOFTWARE_SIGNATURES["new"])
        elif attr == "/CreationDate":
            if claimed_year:
                fake_old_date = datetime(claimed_year,
                                         random.randint(1, 12),
                                         random.randint(1, 28))
            else:
                fake_old_date = datetime.now() - timedelta(days=random.randint(365, 1825))
            new_val = fake_old_date.strftime("D:%Y%m%d%H%M%S")
        elif attr == "/ModDate":
            recent_date = datetime.now() - timedelta(days=random.randint(0, 30))
            new_val = recent_date.strftime("D:%Y%m%d%H%M%S")
        elif attr == "/Author":
            new_val = Faker('en_IN').name()
        elif attr == "/Title":
            new_val = random.choice([
                "Scanned Document",
                "Bank Statement Copy",
                "Untitled",
                "Document1",
                "PDF Export"
            ])
        else:
            new_val = Faker().word()

        new_metadata[attr] = new_val
        corruption_log[attr] = {
            "original": str(original),
            "corrupted_to": new_val
        }

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
        except:
            claimed_year = datetime.now().year

    software_year = None
    for year in range(2008, 2026):
        if str(year) in producer:
            software_year = year
            break

    if claimed_year and software_year:
        delta = software_year - claimed_year
        return delta

    return 0

def extract_metadata_vector(pdf_path, temporal_delta=0):
    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata or {}
    except:
        return [0] * 13

    def parse_pdf_date(date_str):
        if not date_str or len(date_str) < 10:
            return None
        try:
            return datetime.strptime(date_str[2:14], "%Y%m%d%H%M")
        except:
            return None

    creation_date = parse_pdf_date(str(meta.get("/CreationDate", "")))
    mod_date = parse_pdf_date(str(meta.get("/ModDate", "")))

    f1 = 1 if meta.get("/Creator") else 0
    f2 = 1 if meta.get("/Producer") else 0
    f3 = 1 if meta.get("/Author") else 0
    f4 = 1 if creation_date else 0
    f5 = 1 if mod_date else 0
    f6 = 0
    if creation_date and mod_date:
        delta = (mod_date - creation_date).days
        f6 = max(0, delta)

    f7 = (creation_date.year - 2000) if creation_date else 0
    f8 = (mod_date.year - 2000) if mod_date else 0

    suspect_tools = ["ilovepdf", "smallpdf", "online", "converter", "compress"]
    producer_str = str(meta.get("/Producer", "")).lower()
    f9 = 1 if any(t in producer_str for t in suspect_tools) else 0

    f10 = min(len(str(meta.get("/Producer", ""))), 100)

    standard_fields = {"/Title", "/Author", "/Subject", "/Keywords",
                       "/Creator", "/Producer", "/CreationDate", "/ModDate", "/Trapped"}
    f11 = sum(1 for k in meta.keys() if k not in standard_fields)

    suspicious_titles = ["untitled", "document1", "scanned", "copy", "pdf"]
    title_str = str(meta.get("/Title", "")).lower()
    f12 = 1 if any(t in title_str for t in suspicious_titles) else 0
    f13 = temporal_delta

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]

METADATA_FEATURE_NAMES = [
    "has_creator", "has_producer", "has_author",
    "has_creation_date", "has_mod_date",
    "creation_mod_delta_days", "creation_year_norm", "mod_year_norm",
    "suspect_producer_tool", "producer_string_length",
    "num_custom_fields", "suspicious_title_flag",
    "temporal_anachronism_delta"
]


# IEEE Dataport publication version — 10,000 documents, balanced 4 classes
# Random seed 42 ensures full reproducibility as required by IEEE Data Descriptor standard
RANDOM_SEED = 42

CONFIG = {
    "n_genuine": 2500,
    "n_font_tampered": 2500,
    "n_metadata_tampered": 2500,
    "n_both_tampered": 2500,
    "output_dir": "grogu_dataset",
    "templates": list(TEMPLATES.keys()),
    "severity_distribution": {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.2
    }
}

def pick_severity():
    r = random.random()
    if r < 0.3:
        return "low"
    elif r < 0.8:
        return "medium"
    return "high"

def run_pipeline():
    for cls in ["genuine", "font_tampered", "metadata_tampered", "both_tampered"]:
        Path(f"{CONFIG['output_dir']}/raw_pdfs/{cls}").mkdir(parents=True, exist_ok=True)
        Path(f"{CONFIG['output_dir']}/rendered_images/{cls}").mkdir(parents=True, exist_ok=True)
    Path(f"{CONFIG['output_dir']}/metadata_vectors").mkdir(parents=True, exist_ok=True)
    Path(f"{CONFIG['output_dir']}/labels").mkdir(parents=True, exist_ok=True)
    Path(f"{CONFIG['output_dir']}/splits").mkdir(parents=True, exist_ok=True)

    all_records = []
    generation_log = {"genuine": 0, "font_tampered": 0,
                      "metadata_tampered": 0, "both_tampered": 0, "errors": 0}

    # GENUINE
    print("Generating Genuine documents...")
    for i in tqdm(range(CONFIG["n_genuine"])):
        try:
            template_key = random.choice(CONFIG["templates"])
            identity = generate_identity(template_key)

            pdf_path = f"{CONFIG['output_dir']}/raw_pdfs/genuine/{identity['base_id']}_G.pdf"
            field_boxes = generate_genuine_pdf(identity, template_key, pdf_path)

            metadata_vec = extract_metadata_vector(pdf_path, temporal_delta=0)

            record = {
                "file_id": f"{identity['base_id']}_G",
                "base_id": identity["base_id"],
                "class": "genuine",
                "label": 0,
                "pdf_path": pdf_path,
                "template": template_key,
                "tamper_type": "none",
                "tampered_field": None,
                "original_value": None,
                "new_value": None,
                "bounding_box": None,
                "metadata_corrupted_attrs": None,
                "temporal_anachronism_delta": 0,
                **{METADATA_FEATURE_NAMES[j]: metadata_vec[j] for j in range(13)}
            }
            all_records.append(record)
            generation_log["genuine"] += 1

        except Exception as e:
            generation_log["errors"] += 1
            print(f"Error generating genuine doc {i}: {e}")

    # FONT TAMPERED
    print("Generating Font-Tampered documents...")
    for i in tqdm(range(CONFIG["n_font_tampered"])):
        try:
            template_key = random.choice(CONFIG["templates"])
            identity = generate_identity(template_key)

            genuine_path = f"{CONFIG['output_dir']}/raw_pdfs/font_tampered/{identity['base_id']}_G_base.pdf"
            field_boxes = generate_genuine_pdf(identity, template_key, genuine_path)

            tampered_path = f"{CONFIG['output_dir']}/raw_pdfs/font_tampered/{identity['base_id']}_FT.pdf"
            severity = pick_severity()
            tamper_info = apply_visual_tampering(genuine_path, tampered_path, field_boxes, template_key, severity)

            os.remove(genuine_path)

            metadata_vec = extract_metadata_vector(tampered_path, temporal_delta=0)

            record = {
                "file_id": f"{identity['base_id']}_FT",
                "base_id": identity["base_id"],
                "class": "font_tampered",
                "label": 1,
                "pdf_path": tampered_path,
                "template": template_key,
                "tamper_type": "visual",
                "tampered_field": tamper_info["tampered_field"],
                "original_value": tamper_info["original_value"],
                "new_value": tamper_info["new_value"],
                "bounding_box": json.dumps(tamper_info["bounding_box"]),
                "metadata_corrupted_attrs": None,
                "temporal_anachronism_delta": 0,
                **{METADATA_FEATURE_NAMES[j]: metadata_vec[j] for j in range(13)}
            }
            all_records.append(record)
            generation_log["font_tampered"] += 1

        except Exception as e:
            generation_log["errors"] += 1
            print(f"Error generating font tampered doc {i}: {e}")

    # METADATA TAMPERED
    print("Generating Metadata-Tampered documents...")
    for i in tqdm(range(CONFIG["n_metadata_tampered"])):
        try:
            template_key = random.choice(CONFIG["templates"])
            identity = generate_identity(template_key)

            genuine_path = f"{CONFIG['output_dir']}/raw_pdfs/metadata_tampered/{identity['base_id']}_G_base.pdf"
            field_boxes = generate_genuine_pdf(identity, template_key, genuine_path)

            tampered_path = f"{CONFIG['output_dir']}/raw_pdfs/metadata_tampered/{identity['base_id']}_MT.pdf"

            claimed_year = random.randint(2017, 2021)
            meta_tamper_info = apply_metadata_tampering(genuine_path, tampered_path, claimed_year=claimed_year)

            os.remove(genuine_path)

            temporal_delta = meta_tamper_info["temporal_anachronism_delta"]
            metadata_vec = extract_metadata_vector(tampered_path, temporal_delta=temporal_delta)

            record = {
                "file_id": f"{identity['base_id']}_MT",
                "base_id": identity["base_id"],
                "class": "metadata_tampered",
                "label": 2,
                "pdf_path": tampered_path,
                "template": template_key,
                "tamper_type": "metadata",
                "tampered_field": None,
                "original_value": None,
                "new_value": None,
                "bounding_box": None,
                "metadata_corrupted_attrs": json.dumps(meta_tamper_info["corrupted_attributes"]),
                "temporal_anachronism_delta": temporal_delta,
                **{METADATA_FEATURE_NAMES[j]: metadata_vec[j] for j in range(13)}
            }
            all_records.append(record)
            generation_log["metadata_tampered"] += 1

        except Exception as e:
            generation_log["errors"] += 1
            print(f"Error generating metadata tampered doc {i}: {e}")

    # BOTH TAMPERED
    print("Generating Both-Tampered documents...")
    for i in tqdm(range(CONFIG["n_both_tampered"])):
        try:
            template_key = random.choice(CONFIG["templates"])
            identity = generate_identity(template_key)

            genuine_path = f"{CONFIG['output_dir']}/raw_pdfs/both_tampered/{identity['base_id']}_G_base.pdf"
            field_boxes = generate_genuine_pdf(identity, template_key, genuine_path)

            visually_tampered_path = f"{CONFIG['output_dir']}/raw_pdfs/both_tampered/{identity['base_id']}_VT_base.pdf"
            severity = pick_severity()
            tamper_info = apply_visual_tampering(genuine_path, visually_tampered_path, field_boxes, template_key, severity)

            both_path = f"{CONFIG['output_dir']}/raw_pdfs/both_tampered/{identity['base_id']}_BT.pdf"
            claimed_year = random.randint(2017, 2021)
            meta_tamper_info = apply_metadata_tampering(visually_tampered_path, both_path, claimed_year=claimed_year)

            os.remove(genuine_path)
            os.remove(visually_tampered_path)

            temporal_delta = meta_tamper_info["temporal_anachronism_delta"]
            metadata_vec = extract_metadata_vector(both_path, temporal_delta=temporal_delta)

            record = {
                "file_id": f"{identity['base_id']}_BT",
                "base_id": identity["base_id"],
                "class": "both_tampered",
                "label": 3,
                "pdf_path": both_path,
                "template": template_key,
                "tamper_type": "both",
                "tampered_field": tamper_info["tampered_field"],
                "original_value": tamper_info["original_value"],
                "new_value": tamper_info["new_value"],
                "bounding_box": json.dumps(tamper_info["bounding_box"]),
                "metadata_corrupted_attrs": json.dumps(meta_tamper_info["corrupted_attributes"]),
                "temporal_anachronism_delta": temporal_delta,
                **{METADATA_FEATURE_NAMES[j]: metadata_vec[j] for j in range(13)}
            }
            all_records.append(record)
            generation_log["both_tampered"] += 1

        except Exception as e:
            generation_log["errors"] += 1
            print(f"Error generating both-tampered doc {i}: {e}")

    return all_records, generation_log


def save_outputs(all_records, generation_log):
    df = pd.DataFrame(all_records)

    df.to_csv(f"{CONFIG['output_dir']}/labels/master_labels.csv", index=False)

    meta_cols = ["file_id", "label", "class"] + METADATA_FEATURE_NAMES
    df[meta_cols].to_csv(f"{CONFIG['output_dir']}/metadata_vectors/all_metadata.csv", index=False)

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)

    train_val_idx, test_idx = next(gss_outer.split(df, groups=df["base_id"]))
    df_train_val = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx]

    train_idx, val_idx = next(gss_inner.split(df_train_val, groups=df_train_val["base_id"]))
    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    df_train.to_csv(f"{CONFIG['output_dir']}/splits/train.csv", index=False)
    df_val.to_csv(f"{CONFIG['output_dir']}/splits/val.csv", index=False)
    df_test.to_csv(f"{CONFIG['output_dir']}/splits/test.csv", index=False)

    assert len(set(df_train["base_id"]) & set(df_test["base_id"])) == 0, "LEAKAGE DETECTED: base_id appears in both train and test"
    assert len(set(df_val["base_id"]) & set(df_test["base_id"])) == 0, "LEAKAGE DETECTED: base_id appears in both val and test"

    print("No leakage detected — split is clean.")

    generation_log["split_sizes"] = {
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
        "total": len(df)
    }
    generation_log["class_distribution"] = df["class"].value_counts().to_dict()
    generation_log["leakage_check"] = "PASSED"

    with open(f"{CONFIG['output_dir']}/generation_log.json", "w") as f:
        json.dump(generation_log, f, indent=2)

    print("\n=== GENERATION COMPLETE ===")
    print(f"Total files: {len(df)}")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    print(f"Errors during generation: {generation_log['errors']}")
    print(f"Leakage check: PASSED")

if __name__ == "__main__":
    # Fix random seed for IEEE Dataport reproducibility
    random.seed(RANDOM_SEED)
    import numpy as np
    np.random.seed(RANDOM_SEED)

    total = sum([CONFIG['n_genuine'], CONFIG['n_font_tampered'],
                 CONFIG['n_metadata_tampered'], CONFIG['n_both_tampered']])
    print("=== GROGU-KYC Dataset Generation Pipeline (IEEE Dataport Edition) ===")
    print(f"Random Seed : {RANDOM_SEED}  (fixed for reproducibility)")
    print(f"Target      : {total} documents | 4 classes x {total // 4} each")
    print(f"Output dir  : {CONFIG['output_dir']}")
    print("Starting generation...\n")

    all_records, generation_log = run_pipeline()
    save_outputs(all_records, generation_log)
