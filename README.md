# Model-Service
The model-service represents a wrapper service for the URL Phising Detection model. It will offer a REST API to exposes the model to other components and make it scalable. 

## What it does
A POST request can be send to the `/predict` endpoint, after which the model will return whether the URL is likely "Phishing" or "Legitimate". The full API documentation can be found at the `/apidocs` endpoint when running the model-service.

## How to run
The model-service is used in an app frontend, which can be ran using the `compose.yml` in the [operation](https://github.com/REMLA24-Team-5/operation) repository. If you want to run the model-service separately, either use the Dockferfile, then run the following commands

```
docker build -t <tag> .
docker run -it --rm <tag>
```
or run the `model-service.py` by first creating an environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
and then running the Flask app.
```
export FLASK_APP=model-service.py
flask run
```