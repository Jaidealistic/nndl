import os
from pathlib import Path
from dataset_generator import generate_identity, TEMPLATES, generate_genuine_pdf

def generate_samples():
    output_dir = "sample_templates"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Generating template samples...")
    for template_key in TEMPLATES.keys():
        identity = generate_identity(template_key)
        pdf_path = f"{output_dir}/{template_key}_sample.pdf"
        generate_genuine_pdf(identity, template_key, pdf_path)
        print(f"Generated {pdf_path}")

if __name__ == "__main__":
    generate_samples()
    print("Sample generation complete. Please review the PDFs in 'sample_templates' directory.")
