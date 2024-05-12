FROM python:3.10-slim
WORKDIR /root
RUN apt-get update && apt-get install -y && apt-get install -y git
COPY requirements.txt /root/
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["python", "model-service.py"]