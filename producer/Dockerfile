FROM python:3.9-slim
WORKDIR /app

# Install necessary packages including kaggle client
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kaggle

# Create directory for Kaggle API token
# The Kaggle API client expects the token at /root/.kaggle/kaggle.json (when running as root)
RUN mkdir -p /root/.kaggle

# Copy the Kaggle API token from the build context
# IMPORTANT: Ensure 'kaggle.json' is in the 'producer' directory alongside this Dockerfile
COPY kaggle.json /root/.kaggle/kaggle.json

# Set permissions for the Kaggle token
RUN chmod 600 /root/.kaggle/kaggle.json

# Create directory for the dataset
RUN mkdir -p /app/kaggle_dataset

# Download and unzip the dataset
# Replace 'saurabhshahane/seoul-bike-trip-duration-prediction' with your dataset if different
# The -p flag specifies the path to download to. --unzip will extract it.
# This will download and unzip 'SeoulBikeData.csv' into /app/kaggle_dataset/
RUN kaggle datasets download -d saurabhshahane/seoul-bike-trip-duration-prediction -p /app/kaggle_dataset --unzip

COPY producer.py .
# The DATASET_PATH environment variable will be set in docker-compose.yml
# to point to /app/kaggle_dataset/SeoulBikeData.csv
CMD ["python", "producer.py"]