import os
import json
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType

SPARK_MASTER_URL = os.getenv('SPARK_MASTER_URL', "local[*]")
APP_NAME = os.getenv('APP_NAME', "BikeTripModelTrainer")
DATA_BATCHES_DIR = os.getenv('DATA_BATCHES_DIR', "/app/data_batches")
MODELS_DIR = os.getenv('MODELS_DIR', "/app/spark_models")

os.makedirs(MODELS_DIR, exist_ok=True)

spark = SparkSession.builder \
    .appName(APP_NAME) \
    .master(SPARK_MASTER_URL) \
    .config("spark.sql.legacy.setCommandReorder", "true") \
    .getOrCreate()

print(f"Spark session created. AppName: {APP_NAME}, Master: {SPARK_MASTER_URL}")

# Updated feature_cols to include location coordinates
feature_cols = [
    'Distance', 'PLong', 'PLatd', 'DLong', 'DLatd', 'Haversine',
    'Pmonth', 'Pday', 'Phour', 'Pmin', 'PDweek',
    'Temp', 'Precip', 'Wind', 'Humid', 'Solar', 'Snow', 'GroundTemp', 'Dust'
]
label_col = 'Duration'

# All columns expected from CSV that might need type casting
all_numeric_cols_from_csv = ['Duration', 'Distance', 'PLong', 'PLatd', 'DLong', 'DLatd',
                             'Haversine', 'Pmonth', 'Pday', 'Phour', 'Pmin', 'PDweek', 'Dmonth',
                             'Dday', 'Dhour', 'Dmin', 'DDweek', 'Temp', 'Precip', 'Wind',
                             'Humid', 'Solar', 'Snow', 'GroundTemp', 'Dust']


def train_and_save_model(batch_files, model_name_suffix):
    model_identifier = f"model_{model_name_suffix}"
    print(f"\n--- Training {model_identifier} using {len(batch_files)} batch(es) ---")
    full_data_path = [os.path.join(DATA_BATCHES_DIR, f) for f in batch_files]
    
    try:
        df = spark.read.option("header", "true").option("inferSchema", "false").csv(full_data_path)
        
        for col_name in all_numeric_cols_from_csv:
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast(FloatType()))
        
        df = df.na.drop(subset=[label_col])
        
        # Ensure all features defined in feature_cols are present before selection
        # This also defines the order for the VectorAssembler
        current_feature_cols_in_df = [f_col for f_col in feature_cols if f_col in df.columns]
        if len(current_feature_cols_in_df) != len(feature_cols):
            print(f"Warning for {model_identifier}: Mismatch in expected features. Using: {current_feature_cols_in_df}")
            # Potentially problematic if VectorAssembler expects a fixed set later.
            # For this setup, we assume all feature_cols will be present in the CSVs.

        df = df.select(feature_cols + [label_col]) # Use the global feature_cols
        
        print(f"Schema for {model_identifier} after casting and selection:")
        df.printSchema()
        
        imputer_inputs = feature_cols # Use the global feature_cols
        imputer_outputs = [f"{f_col}_imputed" for f_col in imputer_inputs]
        
        imputer = Imputer(inputCols=imputer_inputs, outputCols=imputer_outputs).setStrategy("mean")
        
        assembler_inputs = imputer_outputs
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

        rf = RandomForestRegressor(featuresCol="features", labelCol=label_col, seed=42)
        pipeline = Pipeline(stages=[imputer, assembler, rf])

        print(f"Starting model training for {model_identifier}...")
        trained_pipeline_model = pipeline.fit(df)
        print(f"Model training for {model_identifier} completed.")

        model_path = os.path.join(MODELS_DIR, model_identifier + "_rf")
        trained_pipeline_model.save(model_path)
        print(f"{model_identifier} saved to {model_path}")

        # Save Feature Importances (using the global feature_cols as the basis for names)
        rf_model_from_pipeline = trained_pipeline_model.stages[-1]
        importances = rf_model_from_pipeline.featureImportances
        
        feature_importance_map = {}
        # The importances vector corresponds to the order in `feature_cols` (after imputation)
        for i, feature_name in enumerate(feature_cols): 
             feature_importance_map[feature_name] = float(importances[i])

        sorted_feature_importances = dict(sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True))
        importance_file_path = os.path.join(MODELS_DIR, model_identifier + "_rf_feature_importances.json")
        with open(importance_file_path, 'w') as f_imp:
            json.dump(sorted_feature_importances, f_imp, indent=4)
        print(f"Feature importances for {model_identifier} saved to {importance_file_path}")

    except Exception as e:
        print(f"Error during training/saving {model_identifier}: {e}")
        import traceback
        traceback.print_exc()

# Train models
train_and_save_model(["batch_0.csv"], "1")
train_and_save_model(["batch_0.csv", "batch_1.csv"], "2")
train_and_save_model(["batch_0.csv", "batch_1.csv", "batch_2.csv"], "3")

print("\nAll models trained and saved.")
spark.stop()
print("Spark session stopped.")