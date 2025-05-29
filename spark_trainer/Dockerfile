FROM python:3.9-slim 
# Base Python image. Spark itself is not installed here, as we'll use pyspark library in local mode.
# For a "real" Spark cluster, you'd use a Spark base image or connect to one.

WORKDIR /app

# Install Java (OpenJDK) - PySpark requires Java
RUN apt-get update && \
    apt-get install -y openjdk-11-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY spark_trainer.py .
# data_batches and spark_models will be mounted via volumes

CMD ["python", "spark_trainer.py"]