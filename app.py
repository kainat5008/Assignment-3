# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Load model and scaler from Hugging Face
model_path = hf_hub_download(repo_id="keenu-5008/california-housing-regression", filename="batch_gd_model.pkl")
scaler_path = hf_hub_download(repo_id="keenu-5008/california-housing-regression", filename="scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')  # Frontend will be here

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from frontend
        data = request.json.get('features')
        input_data = np.array(data).reshape(1, -1)
        
        # Preprocess and predict
        scaled_data = scaler.transform(input_data)
        scaled_data = np.c_[np.ones(scaled_data.shape[0]), scaled_data]  # Add bias term
        prediction = model.predict(scaled_data)[0]
        
        return jsonify({"prediction": prediction * 1000})  # Convert to dollar value
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)