import os
from flask import Flask, request, jsonify
from flasgger import Swagger
from joblib import load
import numpy as np
import importlib
pre_process = importlib.import_module('lib-ml.pre_process')

app = Flask(__name__)
swagger = Swagger(app, template_file="swagger.yml")

# Function to fetch the pre-trained model
def fetch_model():
    """
    Fetches the pre-trained model from Google Drive
    """
    model_folder = "/home/resources"
    output = 'model-v1.joblib'
    model = load(os.path.join(model_folder, output))
    return model

def fetch_preprocessing():
    """
    Fetches the pre-processing module from Google Drive
    """
    
    train = [line.strip() for line in open('/home/resources/train.txt', "r", encoding="utf-8").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open('/home/resources/test.txt', "r", encoding="utf-8").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]

    val=[line.strip() for line in open('/home/resources/val.txt', "r", encoding="utf-8").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]

    preprocessor = pre_process.Preprocessing(raw_x_train, raw_y_train, raw_x_test, raw_x_val)

    return preprocessor

# Load the pre-trained model
model = fetch_model()
preprocessor = fetch_preprocessing()

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction based on the URL provided
    """
    # Get the URL from the request
    url = request.json["url"]

    if url:
        # Pre-process the data using lib-ml
        preprocessed_url = preprocessor.process_URL(url)
        preprocessed_url = preprocessed_url.reshape(1, 200, 1) # Reshape the data for the model

        # Make predictions using the pre-trained model
        prediction = model.predict(preprocessed_url, batch_size=1)

        # Convert predicted probabilities to binary labels
        prediction_binary = (np.array(prediction) > 0.5).astype(int)

        if prediction_binary == 1:
            return jsonify({"prediction": "Phishing"})
        else:
            return jsonify({"prediction": "Legitimate"})
    else:
        return jsonify({"error": "No URL found in the request"})

