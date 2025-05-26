import json
import os
import time
import csv
from kafka import KafkaConsumer

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'bike_trips')
GROUP_ID = os.getenv('GROUP_ID', 'bike_trip_consumer_group')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10000))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './data_batches') # Path inside container
NUM_BATCHES_TO_WRITE = int(os.getenv('NUM_BATCHES_TO_WRITE', 3))

os.makedirs(OUTPUT_DIR, exist_ok=True)

consumer = None
while consumer is None:
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset='earliest', # Start reading from the beginning of the topic
            group_id=GROUP_ID,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            consumer_timeout_ms=30000 # Timeout after 30s of no messages
        )
        print("Kafka Consumer connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}. Retrying in 5 seconds...")
        time.sleep(5)


print(f"Consuming from topic: {KAFKA_TOPIC}, Group ID: {GROUP_ID}")
print(f"Batch size: {BATCH_SIZE}, Output directory: {OUTPUT_DIR}")
print(f"Will write a maximum of {NUM_BATCHES_TO_WRITE} batch files.")

batch_data = []
batch_count = 0
headers_written = False
csv_headers = None

try:
    for message_count, message in enumerate(consumer):
        data = message.value
        # print(f"Received message: {data}")

        if not isinstance(data, dict):
            print(f"Skipping non-dict message: {data}")
            continue

        if not csv_headers:
            csv_headers = list(data.keys()) # Infer headers from first valid message

        batch_data.append(data)

        if (message_count + 1) % 100 == 0:
             print(f"Consumer processed {message_count + 1} messages...")

        if len(batch_data) >= BATCH_SIZE:
            if batch_count < NUM_BATCHES_TO_WRITE:
                batch_file_path = os.path.join(OUTPUT_DIR, f"batch_{batch_count}.csv")
                try:
                    with open(batch_file_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_headers)
                        writer.writeheader()
                        writer.writerows(batch_data)
                    print(f"Batch {batch_count} written to {batch_file_path} with {len(batch_data)} records.")
                except Exception as e_write:
                    print(f"Error writing batch {batch_count} to {batch_file_path}: {e_write}")
                
                batch_data = []
                batch_count += 1
            else:
                print(f"Reached NUM_BATCHES_TO_WRITE limit ({NUM_BATCHES_TO_WRITE}). Stopping consumer after this batch if no more messages.")
                # We can break here if we strictly only want NUM_BATCHES_TO_WRITE
                # For now, let it consume remaining messages in the current BATCH_SIZE window
                # then timeout if producer stops.
                # break # Uncomment to stop strictly after NUM_BATCHES_TO_WRITE

        if batch_count >= NUM_BATCHES_TO_WRITE and not batch_data: # All desired batches written and current buffer empty
             print(f"All {NUM_BATCHES_TO_WRITE} batches written. Consumer will now idle or timeout.")
             # break # if you want to stop consuming immediately

    # Write any remaining data if the loop finishes (e.g., due to consumer_timeout_ms)
    if batch_data and batch_count < NUM_BATCHES_TO_WRITE:
        batch_file_path = os.path.join(OUTPUT_DIR, f"batch_{batch_count}.csv")
        try:
            with open(batch_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(batch_data)
            print(f"Final batch {batch_count} written to {batch_file_path} with {len(batch_data)} records.")
        except Exception as e_write:
            print(f"Error writing final batch {batch_count} to {batch_file_path}: {e_write}")

except Exception as e:
    print(f"An error occurred in the consumer: {e}")
finally:
    if consumer:
        consumer.close()
        print("Kafka consumer closed.")
    print(f"Consumer finished. Total batches written: {batch_count}")