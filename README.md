# Assignment-3
This is my assignment 3 for Machine Learning


# Task1 Button 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kainat5008/Assignment-3/blob/main/ML_TASK1_A3.ipynb)

# Task2 

## Hugging Face Model
[View Model on Hugging Face](https://huggingface.co/keenu-5008/california-housing-regression)

---
tags:
- regression
- scikit-learn
- gradient-descent
---

# California Housing Price Prediction (Regression)

This model predicts california housing prices using **Batch Gradient Descent** with L2 regularization.

## Usage
```python
import joblib
from huggingface_hub import hf_hub_download

# Download files
model_path = hf_hub_download(repo_id="your-username/california-housing-regression", filename="batch_gd_model.pkl")
scaler_path = hf_hub_download(repo_id="your-username/california-housing-regression", filename="scaler.pkl")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Example prediction
import numpy as np
new_data = np.array([[0.1, 20.0, 5.0, ...]])  # Replace with your data
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)
print(prediction)
```

# Task3
## Inference Script

### How to Run
1. Install dependencies:
   ```bash
   pip install numpy joblib huggingface_hub
   ```
2. Download `inference.py` and run:
   ```bash
   python inference.py
   ```
3. Enter 8 features when prompted (e.g., `0.1 20.0 5.0 1.0 500.0 6.0 40.0 -122.0`).

### Expected Output
```
=== Boston Housing Price Predictor ===
Enter features separated by spaces: 0.1 20.0 5.0 1.0 500.0 6.0 40.0 -122.0
Predicted Price: $250000.00
```
# Task4

## Weights & Biases Integration
Training metrics were tracked using W&B. View the live dashboard here:  
[![W&B Dashboard](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/kainatkhalid-5008-fast-nuces/california-housing-regression)

Key tracked metrics:
- Training loss
- Test loss
- Hyperparameters

  
## Task 5: Web UI

### How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Flask app:
   ```bash
   python app.py
   ```
3. Access at `http://localhost:5000`

### Files Submitted
- `app.py` (Flask backend)
- `templates/index.html` (Frontend)
- `requirements.txt` (Dependencies)
  ```
