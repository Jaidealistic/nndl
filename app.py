import os
import sys
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add src/model to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'model'))
from predict import GROGUPredictor

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
DEMO_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'demo_samples')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "src", "model", "grogu_model.pth")
predictor = GROGUPredictor(MODEL_PATH)

SAMPLE_CATALOG = [
    {
        "id": "sample_genuine_kyc.pdf",
        "title": "Standard Income Certificate",
        "category": "Genuine (Clean)",
        "expected": "Genuine",
        "badge_color": "emerald",
        "description": "Authentic government-issued format with pristine typography and verified generation timestamps."
    },
    {
        "id": "sample_font_tampered.pdf",
        "title": "Salary Slip with Modified CTC",
        "category": "Optical Anomaly",
        "expected": "Font Tampered",
        "badge_color": "amber",
        "description": "Gross pay field altered using non-standard system font; kerning and anti-aliasing diverge."
    },
    {
        "id": "sample_metadata_tampered.pdf",
        "title": "Bank Statement (Editor Signature)",
        "category": "Metadata Forgery",
        "expected": "Metadata Tampered",
        "badge_color": "rose",
        "description": "Visually pristine layout, but internal /Producer reveals unauthorized consumer PDF tool."
    },
    {
        "id": "sample_both_tampered.pdf",
        "title": "Forged Financial Dossier",
        "category": "Compound Attack",
        "expected": "Both Tampered",
        "badge_color": "purple",
        "description": "Dual attack: Optical field splicing combined with temporal creation anachronism."
    }
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": "GROGU-v2.0-Bimodal",
        "device": str(predictor.device),
        "classes": list(predictor.classes.values())
    })

@app.route('/samples', methods=['GET'])
def get_samples():
    return jsonify({"samples": SAMPLE_CATALOG})

@app.route('/predict_sample/<sample_id>', methods=['GET'])
def predict_sample(sample_id):
    safe_name = secure_filename(sample_id)
    sample_path = os.path.join(DEMO_SAMPLES_DIR, safe_name)
    if not os.path.exists(sample_path):
        return jsonify({"error": f"Sample '{sample_id}' not found"}), 404
    
    heatmap_path = os.path.join(UPLOAD_FOLDER, f"heatmap_sample_{safe_name}.jpg")
    try:
        result = predictor.predict_and_explain(sample_path, heatmap_path)
        result["filename"] = safe_name
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Please select a PDF."}), 400
        
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid format. Only PDF files are supported."}), 400
        
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)
    
    heatmap_path = os.path.join(UPLOAD_FOLDER, f"heatmap_{filename}.jpg")
    
    try:
        result = predictor.predict_and_explain(pdf_path, heatmap_path)
        result["filename"] = filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Forensic analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
