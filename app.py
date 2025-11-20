from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
try:
    model_path = 'models/savedmodel.pth'
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image_file):
    try:
        # Open image
        img = Image.open(image_file)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 64x64 (Olivetti faces format)
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img).flatten() / 255.0
        
        return img_array.reshape(1, -1)
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image uploaded'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No image selected'
        }), 400
    
    try:
        # Preprocess image
        img_data = preprocess_image(file)
        
        if img_data is None:
            return jsonify({
                'success': False,
                'error': 'Failed to preprocess image'
            }), 400
        
        # Make prediction
        prediction = model.predict(img_data)[0]
        prediction_proba = model.predict_proba(img_data)[0]
        confidence = float(np.max(prediction_proba) * 100)
        
        return jsonify({
            'success': True,
            'predicted_class': int(prediction),
            'confidence': round(confidence, 2),
            'message': f'Predicted Person: {int(prediction)}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
