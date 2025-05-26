FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY consumer.py .
# data_batches directory will be mounted via volume
CMD ["python", "consumer.py"]