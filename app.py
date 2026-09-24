import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Add predict.py to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'model'))
from predict import GROGUPredictor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "src/model/grogu_model.pth"
predictor = GROGUPredictor(MODEL_PATH)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)
    
    heatmap_path = os.path.join(UPLOAD_FOLDER, f"heatmap_{filename}.jpg")
    
    try:
        result = predictor.predict_and_explain(pdf_path, heatmap_path)
        
        # Encode heatmap to base64
        with open(heatmap_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        result["heatmap_base64"] = f"data:image/jpeg;base64,{encoded_string}"
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
