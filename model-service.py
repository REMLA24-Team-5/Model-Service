from flask import Flask, request, jsonify
from lib_ml import Preprocessing
import gdown
from joblib import load
import numpy as np

app = Flask(__name__)

# Function to fetch the pre-trained model
def fetch_model():
    """
    Fetches the pre-trained model from Google Drive
    """
    model_url = "https://drive.google.com/file/d/185n3q-K-l3eiFwiThouljU_j9rYDugIX"
    output = 'model/model.joblib'
    model = load(gdown.download(model_url, output, quiet=False))
    return model

# Load the pre-trained model
model = fetch_model()

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    """
    # Get the URL from the request
    url = request.json

    if url:
        # Pre-process the data using lib-ml
        preprocessed_url = Preprocessing.process_URL(url)

        # Make predictions using the pre-trained model
        prediction = model.predict(preprocessed_url)

        # Convert predicted probabilities to binary labels
        prediction_binary = (np.array(prediction) > 0.5).astype(int)

        return jsonify({"prediction": prediction_binary.tolist()})
    else:
        return jsonify({"error": "No URL found in the request"})

if __name__ == '__main__':
    app.run(debug=True)
