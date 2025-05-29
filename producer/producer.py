import csv
import json
import time
import random
import os
from kafka import KafkaProducer # Ini adalah impor yang benar untuk producer

# TIDAK ADA impor pyspark di sini

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:29092')
# DATASET_PATH akan diambil dari environment variable yang diatur di docker-compose.yml
DATASET_PATH = os.getenv('DATASET_PATH', '/app/kaggle_dataset/For_modeling.csv')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'bike_trips')
MAX_ROWS_TO_SEND = int(os.getenv('MAX_ROWS_TO_SEND', 30005)) 

producer = None
while producer is None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print(f"Kafka Producer connected successfully to {KAFKA_BROKER}.")
    except Exception as e:
        print(f"Failed to connect to Kafka ({KAFKA_BROKER}): {e}. Retrying in 5 seconds...")
        time.sleep(5)

print(f"Reading dataset from: {DATASET_PATH}")
print(f"Sending to Kafka topic: {KAFKA_TOPIC} on broker: {KAFKA_BROKER}")
print(f"Will send a maximum of {MAX_ROWS_TO_SEND} rows.")

try:
    with open(DATASET_PATH, 'r', encoding='utf-8') as file: # Atau encoding='iso-8859-1' jika perlu untuk dataset spesifik
        csv_reader = csv.DictReader(file)
        headers = csv_reader.fieldnames
        print(f"CSV Headers: {headers}")
        
        # Kolom yang diharapkan sebagai numerik (sesuai dengan koreksi sebelumnya)
        numeric_cols = ['Duration', 'Distance', 'PLong', 'PLatd', 'DLong', 'DLatd',
                        'Haversine', 'Pmonth', 'Pday', 'Phour', 'Pmin', 'PDweek', 'Dmonth',
                        'Dday', 'Dhour', 'Dmin', 'DDweek', 'Temp', 'Precip', 'Wind',
                        'Humid', 'Solar', 'Snow', 'GroundTemp', 'Dust']
        
        rows_sent_count = 0
        for i, row in enumerate(csv_reader):
            if rows_sent_count >= MAX_ROWS_TO_SEND:
                print(f"Reached MAX_ROWS_TO_SEND limit of {MAX_ROWS_TO_SEND}. Stopping producer.")
                break
            try:
                # Konversi tipe data dasar
                processed_row = {}
                for col_name, value in row.items():
                    if col_name in numeric_cols:
                        try:
                            processed_row[col_name] = float(value) if value else 0.0 # Handle empty strings as 0.0
                        except ValueError:
                            print(f"Warning: Could not convert column '{col_name}' value '{value}' to float for row {i}. Using 0.0.")
                            processed_row[col_name] = 0.0
                    else:
                        processed_row[col_name] = value # Biarkan string apa adanya

                producer.send(KAFKA_TOPIC, processed_row)
                rows_sent_count += 1

                if rows_sent_count % 1000 == 0: # Log setiap 1000 pesan
                    print(f"Sent message {rows_sent_count}: {processed_row.get('Duration', 'N/A')}")
                
                # Simulate streaming dengan jeda acak
                time.sleep(random.uniform(0.0005, 0.005)) # Jeda kecil

            except Exception as e_row:
                print(f"Error processing/sending row {i}: {row}. Error: {e_row}")
        
        producer.flush()
        print(f"Finished sending {rows_sent_count} data rows.")

except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    print("Please ensure the dataset is downloaded correctly by the Kaggle API in the Docker build process,")
    print("and that DATASET_PATH in docker-compose.yml points to the correct location inside the container.")
except Exception as e:
    print(f"An error occurred in the producer: {e}")
    import traceback
    traceback.print_exc()
finally:
    if producer:
        producer.close()
        print("Kafka producer closed.")