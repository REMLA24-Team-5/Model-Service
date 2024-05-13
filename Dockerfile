FROM python:3.10-slim
WORKDIR /root
RUN apt-get update && apt-get install -y && apt-get install -y git
COPY requirements.txt /root/
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8000
ENV FLASK_APP=model-service.py
CMD ["flask", "run", "--host=0.0.0.0"]
