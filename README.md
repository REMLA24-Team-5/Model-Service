# Model-service
The model-service represents a wrapper service for the URL Phishing Detection model. It will offer a REST API to exposes the model to other components and make it scalable. 

## What it does
A POST request can be send to the `/predict` endpoint, after which the model will return whether the URL is likely "Phishing" or "Legitimate". The full API documentation can be found at the `/apidocs` endpoint when running the model-service.

## How to run
The model-service is queried through in an app-frontend, which can be ran using the `compose.yml` in the [operation](https://github.com/REMLA24-Team-5/operation) repository. The model-service is currently not supposed to be run in isolation, since it uses a volume mount for the trained model. This model is stored in the operation repository, meaning that it will not find the model when building the model-service in this repository. This could be bypassed by manually downloading the trained model and training files (`volume` folder in [operation](https://github.com/REMLA24-Team-5/operation)). After that make sure to update the file locations in the `model-service.py` (i.e. in line 22, 32, 36, and 39). Having updated this, the model-service can be build and run using the Dockerfile:
```
docker build -t <tag> .
docker run -it --rm -p8000:8000 <tag>
```
or alternatively run the `model-service.py` by first creating an environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
and then running the Flask app.
```
export FLASK_APP=model-service.py
export FLASK_RUN_PORT=8000
flask run
```