FROM python:3.9-slim

WORKDIR /app

# Install Java (OpenJDK 17) - PySpark requires Java
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update JAVA_HOME to point to OpenJDK 17
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY spark_trainer.py .
# data_batches and spark_models will be mounted via volumes

CMD ["python", "spark_trainer.py"]