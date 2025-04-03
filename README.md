# Assignment-3
This is my assignment 3 for Machine Learning


# Task1 Button 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kainat5008/Assignment-3/blob/main/ML_TASK1_A3.ipynb)

# Task2 
---
tags:
- regression
- scikit-learn
- gradient-descent
---

# california Housing Price Prediction (Regression)

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

## Hugging Face Model
[View Model on Hugging Face](https://huggingface.co/keenu-5008/california-housing-regression)
