import os
import json # Untuk menyimpan feature importance
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
print(f"Reading batches from: {DATA_BATCHES_DIR}")
print(f"Saving models to: {MODELS_DIR}")

feature_cols = ['Distance', 'Haversine', 'Temp', 'Wind', 'Humid', 'Solar',
                'Pmonth', 'Pday', 'Phour', 'Pmin', 'PDweek',
                'Precip', 'Snow', 'GroundTemp', 'Dust']
label_col = 'Duration'

# Koreksi nama kolom: PickupLong -> PLong, DropoffLong -> DLong
numeric_cols_from_producer = ['Duration', 'Distance', 'PLong', 'PLatd', 'DLong', 'DLatd', # Koreksi di sini
                              'Haversine', 'Pmonth', 'Pday', 'Phour', 'Pmin', 'PDweek', 'Dmonth',
                              'Dday', 'Dhour', 'Dmin', 'DDweek', 'Temp', 'Precip', 'Wind',
                              'Humid', 'Solar', 'Snow', 'GroundTemp', 'Dust']


def train_and_save_model(batch_files, model_name_suffix):
    model_identifier = f"model_{model_name_suffix}" # e.g. model_1
    print(f"\n--- Training {model_identifier} using {len(batch_files)} batch(es) ---")
    full_data_path = [os.path.join(DATA_BATCHES_DIR, f) for f in batch_files]
    print(f"Reading data from: {full_data_path}")

    try:
        df = spark.read.option("header", "true").option("inferSchema", "false").csv(full_data_path)
        
        for col_name in numeric_cols_from_producer:
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast(FloatType()))
            # else:
                # print(f"Warning: Column {col_name} not found in DataFrame for {model_identifier}")
        
        df = df.na.drop(subset=[label_col])
        
        # Select only features that are actually present in the dataframe after reading CSV
        # This handles cases where some columns might be missing in certain CSVs (though unlikely here)
        available_feature_cols = [f_col for f_col in feature_cols if f_col in df.columns]
        if len(available_feature_cols) != len(feature_cols):
            print(f"Warning: Not all defined features found for {model_identifier}. Using: {available_feature_cols}")

        df = df.select(available_feature_cols + [label_col])
        
        print(f"Schema for {model_identifier} after casting and selection:")
        df.printSchema()
        # df.show(5, truncate=False) # Can be verbose
        
        imputer_inputs = available_feature_cols
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

        # --- Save Feature Importances ---
        rf_model_from_pipeline = trained_pipeline_model.stages[-1] # RandomForestRegressionModel
        importances = rf_model_from_pipeline.featureImportances

        # The assembler_inputs are the names of the features *before* "_imputed" was added by imputer
        # but these are the names that went into the vector assembler (after imputation).
        # So, the `available_feature_cols` are the original names corresponding to the vector.
        
        # The VectorAssembler stage (trained_pipeline_model.stages[-2]) used `assembler_inputs`
        # which are `imputer_outputs` (e.g., 'Distance_imputed', 'Temp_imputed').
        # We want to map back to original feature names: `available_feature_cols`.
        
        feature_importance_map = {}
        for i, feature_name in enumerate(available_feature_cols): # original names
             feature_importance_map[feature_name] = float(importances[i])

        # Sort by importance
        sorted_feature_importances = dict(sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True))

        importance_file_path = os.path.join(MODELS_DIR, model_identifier + "_rf_feature_importances.json")
        with open(importance_file_path, 'w') as f_imp:
            json.dump(sorted_feature_importances, f_imp, indent=4)
        print(f"Feature importances for {model_identifier} saved to {importance_file_path}")

    except Exception as e:
        print(f"Error during training/saving {model_identifier}: {e}")
        import traceback
        traceback.print_exc()

# --- Model 1: Batch 0 ---
train_and_save_model(["batch_0.csv"], "1")

# --- Model 2: Batch 0 + Batch 1 ---
train_and_save_model(["batch_0.csv", "batch_1.csv"], "2")

# --- Model 3: Batch 0 + Batch 1 + Batch 2 ---
train_and_save_model(["batch_0.csv", "batch_1.csv", "batch_2.csv"], "3")

print("\nAll models trained and saved.")
spark.stop()
print("Spark session stopped.")