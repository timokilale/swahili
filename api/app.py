# api/app.py
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from swahili_preprocessor import SwahiliPreprocessor

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
feature_extractor = None
preprocessor = None

def load_best_model():
    """HARDCODED: Always load Bi-LSTM model"""
    global model, feature_extractor, preprocessor

    print("🧠 HARDCODED: Loading Bi-LSTM model...")

    # Always load Bi-LSTM model
    return load_bilstm_model()

def load_traditional_model():
    """Load traditional ML model (Random Forest, SVM, Logistic Regression)"""
    global model, feature_extractor, preprocessor

    try:
        # Load model
        model = joblib.load('models/best_model.pkl')
        print("✅ Traditional ML model loaded successfully")

        # Load feature extractor
        try:
            feature_extractor = joblib.load('models/feature_extractor.pkl')
            print("✅ Feature extractor loaded successfully")
        except:
            print("⚠️ Feature extractor not found, creating new instance")
            import sys
            sys.path.append('../src')
            from feature_extractor import SwahiliFeatureExtractor
            feature_extractor = SwahiliFeatureExtractor()

        # Initialize preprocessor
        preprocessor = SwahiliPreprocessor()
        print("✅ Preprocessor initialized")

        return True

    except Exception as e:
        print(f"❌ Error loading traditional model: {e}")
        return False

def load_bilstm_model():
    """Load Bi-LSTM model"""
    global model, feature_extractor, preprocessor

    try:
        import torch
        import sys
        sys.path.append('../src')
        from bilstm_model import BiLSTMClassifier

        # Load model checkpoint
        checkpoint = torch.load('models/bilstm_best_model.pth', map_location='cpu')
        vocab = checkpoint['vocab']

        # Initialize model
        model_instance = BiLSTMClassifier(len(vocab))
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        model_instance.eval()

        # Create wrapper for consistent interface
        class BiLSTMWrapper:
            def __init__(self, model, vocab):
                self.model = model
                self.vocab = vocab
                self.preprocessor = SwahiliPreprocessor()

            def predict_proba(self, X):
                # Convert text to sequence
                sequences = []
                for text in X:
                    processed = self.preprocessor.preprocess(text[0] if isinstance(text, list) else text)
                    sequence = self.text_to_sequence(processed)
                    sequences.append(sequence)

                # Predict
                with torch.no_grad():
                    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
                    outputs = self.model(sequences_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    return probabilities.numpy()

            def text_to_sequence(self, text, max_length=128):
                text = text.lower()
                words = text.split()
                sequence = []
                for word in words[:max_length]:
                    if word in self.vocab:
                        sequence.append(self.vocab[word])
                    else:
                        sequence.append(self.vocab['<UNK>'])
                while len(sequence) < max_length:
                    sequence.append(self.vocab['<PAD>'])
                return sequence[:max_length]

        model = BiLSTMWrapper(model_instance, vocab)
        feature_extractor = None  # Bi-LSTM doesn't use feature extractor
        preprocessor = SwahiliPreprocessor()

        print("✅ Bi-LSTM model loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Error loading Bi-LSTM model: {e}")
        print("🔄 Falling back to traditional ML model...")
        return load_traditional_model()

def load_bert_model():
    """Load BERT model (placeholder for future implementation)"""
    print("⚠️ BERT model loading not implemented yet")
    print("🔄 Falling back to traditional ML model...")
    return load_traditional_model()

# Alias for backward compatibility
load_model = load_best_model

def extract_features_for_prediction(message):
    """Extract features for a single message (only for traditional ML models)"""
    try:
        # Check if we're using a traditional ML model that needs feature extraction
        if feature_extractor is None:
            return None  # Bi-LSTM doesn't use feature extraction

        # Extract linguistic features
        features = preprocessor.extract_basic_features(message)

        # Load training feature columns to ensure consistency
        training_features = pd.read_csv('data/processed/features.csv')

        # Create feature DataFrame
        feature_df = pd.DataFrame([features])

        # Ensure all training columns are present
        # Get missing columns
        missing_cols = [col for col in training_features.columns if col not in feature_df.columns]

        # Add all missing columns at once using pd.concat
        if missing_cols:
            missing_df = pd.DataFrame(0, index=feature_df.index, columns=missing_cols)
            feature_df = pd.concat([feature_df, missing_df], axis=1)

        # Reorder columns to match training data
        feature_df = feature_df[training_features.columns]

        return feature_df

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_message(message):
    """Universal prediction function that works with any model type"""
    try:
        if feature_extractor is None:
            # Bi-LSTM model - direct text input
            probabilities = model.predict_proba([message])
            confidence = float(probabilities[0][1])  # Probability of scam class
            is_scam = confidence > 0.5
        else:
            # Traditional ML model - needs feature extraction
            features = extract_features_for_prediction(message)
            if features is None:
                return None, None

            probabilities = model.predict_proba(features)
            confidence = float(probabilities[0][1])  # Probability of scam class
            is_scam = confidence > 0.5

        return is_scam, confidence

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

@app.route('/')
def home():
    """Home page with simple interface"""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🛡️ Swahili Scam Detector</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            textarea {
                width: 100%;
                height: 120px;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                resize: vertical;
                box-sizing: border-box;
            }
            button {
                background: #667eea;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                margin-top: 15px;
            }
            button:hover {
                background: #5a6fd8;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 10px;
                display: none;
            }
            .scam {
                background: #fee;
                border: 2px solid #dc3545;
                color: #721c24;
            }
            .safe {
                background: #efe;
                border: 2px solid #28a745;
                color: #155724;
            }
            .examples {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .example {
                background: white;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                cursor: pointer;
                border-left: 4px solid #667eea;
            }
            .example:hover {
                background: #f0f0f0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Swahili Scam Detector</h1>
            <p class="subtitle">       </p>
            
            <textarea id="messageInput" placeholder="Andika ujumbe hapa / Enter message here...
Mfano: Hongera! Umeshinda TSH 1,000,000. Piga *123# kuconfirm..."></textarea>
            
            <button onclick="checkMessage()">🔍 Angalia Ujumbe / Check Message</button>
            
            <div id="result" class="result">
                <h3 id="resultTitle"></h3>
                <p id="resultText"></p>
                <p id="confidence"></p>
            </div>
        </div>

        <script>
            async function checkMessage() {
                const message = document.getElementById('messageInput').value.trim();
                
                if (!message) {
                    alert('Tafadhali andika ujumbe kwanza / Please enter a message first');
                    return;
                }
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayResult(result);
                    } else {
                        alert('Kuna hitilafu: ' + result.error);
                    }
                    
                } catch (error) {
                    alert('Kuna hitilafu imetokea: ' + error.message);
                }
            }
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                const resultTitle = document.getElementById('resultTitle');
                const resultText = document.getElementById('resultText');
                const confidence = document.getElementById('confidence');
                
                if (result.is_scam) {
                    resultDiv.className = 'result scam';
                    resultTitle.textContent = '⚠️ Ujumbe wa Udanganyifu / Scam Message Detected!';
                    resultText.textContent = 'Tahadhari! Ujumbe huu unaonekana kuwa wa udanganyifu.';
                } else {
                    resultDiv.className = 'result safe';
                    resultTitle.textContent = '✅ Ujumbe Salama / Safe Message';
                    resultText.textContent = 'Ujumbe huu unaonekana kuwa salama.';
                }
                
                confidence.textContent = `Uhakika / Confidence: ${(result.confidence * 100).toFixed(1)}%`;
                resultDiv.style.display = 'block';
            }
            
            function useExample(element) {
                document.getElementById('messageInput').value = element.textContent.trim();
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/predict', methods=['POST'])
def predict_scam():
    """API endpoint for scam prediction - works with any model type"""
    try:
        # Get message from request
        data = request.json
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Make prediction using universal function
        is_scam, confidence = predict_message(message)

        if is_scam is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Prepare response
        response = {
            'message': message,
            'is_scam': bool(is_scam),
            'confidence': float(confidence),
            'scam_probability': float(confidence if is_scam else 1 - confidence),
            'safe_probability': float(1 - confidence if is_scam else confidence),
            'model_type': 'Bi-LSTM' if feature_extractor is None else 'Traditional ML'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """API status endpoint"""
    model_type = 'None'
    if model is not None:
        model_type = 'Bi-LSTM' if feature_extractor is None else 'Traditional ML'

    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'model_type': model_type,
        'version': '2.0.0 - Smart Model Selection'
    })

@app.route('/api/test')
def test_api():
    """Test endpoint with sample predictions"""
    test_messages = [
        "Hongera! Umeshinda TSH 1,000,000. Piga *150*00# sasa kuconfirm.",
        "Habari za asubuhi. Mkutano wetu ni saa 2 jioni leo.",
        "URGENT! Tuma pesa haraka kwa namba hii 0712345678.",
        "Asante kwa huduma nzuri. Tuonane kesho asubuhi."
    ]
    
    results = []
    for message in test_messages:
        try:
            features = extract_features_for_prediction(message)
            if features is not None:
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]
                
                results.append({
                    'message': message,
                    'is_scam': bool(prediction == 1),
                    'confidence': float(max(probability))
                })
            else:
                results.append({
                    'message': message,
                    'error': 'Feature extraction failed'
                })
        except Exception as e:
            results.append({
                'message': message,
                'error': str(e)
            })
    
    return jsonify({
        'test_results': results,
        'total_tests': len(test_messages)
    })

if __name__ == '__main__':
    print("🚀 Starting Swahili Scam Detection API...")
    print("=" * 50)
    
    # Load model
    if load_model():
        print("✅ All components loaded successfully!")
        print("\n🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🔗 API endpoint: http://localhost:5000/api/predict")
        print("🧪 Test endpoint: http://localhost:5000/api/test")
        print("\n⏹️  Press Ctrl+C to stop the server")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please run model training first.")
        print("💡 Run: python src/model_trainer.py")