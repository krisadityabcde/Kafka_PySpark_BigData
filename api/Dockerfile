FROM python:3.9-slim

WORKDIR /app

# Install Java (OpenJDK) - PySpark requires Java
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
# spark_models will be mounted via volume

CMD ["python", "api.py"]