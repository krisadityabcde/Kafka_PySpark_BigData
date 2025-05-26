import os
import json
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, FloatType
# from datetime import date # Not strictly needed if PDweek is provided by user

app = Flask(__name__)

SPARK_MASTER_URL = os.getenv('SPARK_MASTER_URL', "local[*]")
APP_NAME = os.getenv('APP_NAME', "BikeTripPredictionAPI")
MODELS_DIR = os.getenv('MODELS_DIR', "/app/spark_models")

spark = SparkSession.builder \
    .appName(APP_NAME) \
    .master(SPARK_MASTER_URL) \
    .config("spark.sql.legacy.setCommandReorder", "true") \
    .getOrCreate()

print("Flask API: Spark session created.")

# Load models
loaded_models = {}
model_versions = ["1", "2", "3"]
for version in model_versions:
    model_path = os.path.join(MODELS_DIR, f"model_{version}_rf")
    if os.path.exists(model_path):
        try:
            loaded_models[version] = PipelineModel.load(model_path)
            print(f"Flask API: Loaded model 'model_{version}' from {model_path}")
        except Exception as e:
            print(f"Flask API: Error loading model 'model_{version}' from {model_path}: {e}")
            loaded_models[version] = None
    else:
        print(f"Flask API: Model path {model_path} for 'model_{version}' not found.")
        loaded_models[version] = None

# Updated list of features the model expects for prediction
# Must match feature_cols in spark_trainer.py
PREDICTION_FEATURE_COLS = [
    'Distance', 'PLong', 'PLatd', 'DLong', 'DLatd', 'Haversine',
    'Pmonth', 'Pday', 'Phour', 'Pmin', 'PDweek',
    'Temp', 'Precip', 'Wind', 'Humid', 'Solar', 'Snow', 'GroundTemp', 'Dust'
]
prediction_schema_fields = [StructField(col_name, FloatType(), True) for col_name in PREDICTION_FEATURE_COLS]
prediction_schema = StructType(prediction_schema_fields)

def validate_and_prepare_single_input(data_dict):
    """Validates input data and prepares it in the correct order for the model."""
    input_values = []
    missing_features = []
    invalid_features = {}

    for col_name in PREDICTION_FEATURE_COLS:
        val = data_dict.get(col_name)
        if val is None:
            missing_features.append(col_name)
            input_values.append(0.0) # Default for missing, align with imputation strategy if possible
        else:
            try:
                input_values.append(float(val))
            except ValueError:
                invalid_features[col_name] = val
                input_values.append(0.0) # Default for invalid
    
    if invalid_features:
        raise ValueError(f"Invalid non-numeric values for features: {invalid_features}")
    if missing_features:
        print(f"Warning: Missing features in input, defaulted to 0.0: {missing_features}")
        # Depending on strictness, could raise error here too

    return tuple(input_values), missing_features


# --- Endpoint 1: Predict Duration ---
@app.route('/predict/duration/<model_version_num>', methods=['POST'])
def predict_duration(model_version_num):
    if model_version_num not in loaded_models or loaded_models[model_version_num] is None:
        return jsonify({"error": f"Model 'model_{model_version_num}' not loaded or not found."}), 404
    model_to_use = loaded_models[model_version_num]
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_tuple, missing_features = validate_and_prepare_single_input(data)
        input_df = spark.createDataFrame([input_tuple], schema=prediction_schema)
        
        prediction_result = model_to_use.transform(input_df)
        predicted_duration_value = prediction_result.select("prediction").first()[0]
        
        return jsonify({
            "model_version_used": f"model_{model_version_num}",
            "input_features": data,
            "predicted_duration": predicted_duration_value,
            "missing_features_defaulted": missing_features if missing_features else None
        })
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Error during prediction with model_{model_version_num}: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# --- Endpoint 2: Sensitivity Analysis ---
@app.route('/analyze/sensitivity/<model_version_num>', methods=['POST'])
def analyze_sensitivity(model_version_num):
    if model_version_num not in loaded_models or loaded_models[model_version_num] is None:
        return jsonify({"error": f"Model 'model_{model_version_num}' not loaded."}), 404
    model_to_use = loaded_models[model_version_num]

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No input payload provided"}), 400

        base_features = payload.get('base_features')
        variable_feature_name = payload.get('variable_feature_name')
        variation_values = payload.get('variation_values')

        if not all([base_features, variable_feature_name, variation_values]):
            return jsonify({"error": "Missing 'base_features', 'variable_feature_name', or 'variation_values' in payload."}), 400
        if variable_feature_name not in PREDICTION_FEATURE_COLS:
            return jsonify({"error": f"Invalid 'variable_feature_name': {variable_feature_name}. Not a model feature."}), 400
        if not isinstance(variation_values, list):
            return jsonify({"error": "'variation_values' must be a list."}), 400

        results = []
        prediction_rows = []

        for var_val in variation_values:
            current_features = base_features.copy()
            current_features[variable_feature_name] = var_val
            
            # Validate and prepare this specific set of features
            # We collect all rows first to make one Spark DataFrame for efficiency
            try:
                input_tuple, _ = validate_and_prepare_single_input(current_features)
                prediction_rows.append(input_tuple)
            except ValueError as ve: # Catch validation errors for individual variations
                 results.append({
                    "varied_feature": variable_feature_name,
                    "value": var_val,
                    "error": str(ve),
                    "predicted_duration": None
                })

        if not prediction_rows: # All variations had errors or no variations
            if results: # only errors
                 return jsonify({"analysis_results": results})
            return jsonify({"error": "No valid variations to process after validation."}), 400


        # Create a single DataFrame for all valid variations
        input_df = spark.createDataFrame(prediction_rows, schema=prediction_schema)
        predictions_df = model_to_use.transform(input_df)
        predicted_durations = [row.prediction for row in predictions_df.select("prediction").collect()]

        # Combine predictions with their corresponding variation values
        # This assumes prediction_rows and predicted_durations are in the same order
        # and that variation_values was used to generate prediction_rows.
        # We need to map back carefully if some variations failed validation earlier.
        
        # Rebuild results, including successful predictions
        final_results = []
        valid_variation_idx = 0
        for original_idx, var_val in enumerate(variation_values):
            # Check if this variation was one that passed validation and was added to prediction_rows
            # This is a bit tricky; easier if we store the original varied feature dict
            # For now, let's assume if it's not an error, it's a success.
            # A more robust way would be to pass the full varied feature set and match.
            
            # Simplified: Assume order is maintained for successfully processed items
            # This needs careful handling if `validate_and_prepare_single_input` for a variation fails
            # and that variation is skipped in `prediction_rows`.
            # The current `validate_and_prepare_single_input` raises ValueError, so loop would break.
            # Let's adjust the loop to handle individual validation failures more gracefully for this endpoint.
            
            # Re-looping for clarity and correct association:
            current_features_for_this_var = base_features.copy()
            current_features_for_this_var[variable_feature_name] = var_val
            
            try:
                # Re-validate to know if it was part of the batch sent to Spark
                validate_and_prepare_single_input(current_features_for_this_var) 
                # If it passed validation, it should have a corresponding prediction
                if valid_variation_idx < len(predicted_durations):
                    final_results.append({
                        "varied_feature": variable_feature_name,
                        "value": var_val,
                        "predicted_duration": predicted_durations[valid_variation_idx],
                        "base_features_snapshot": current_features_for_this_var # For reference
                    })
                    valid_variation_idx += 1
                else: # Should not happen if logic is correct
                     final_results.append({
                        "varied_feature": variable_feature_name,
                        "value": var_val,
                        "error": "Prediction missing, data inconsistency.",
                        "predicted_duration": None
                    })
            except ValueError as ve_inner: # This variation failed validation
                final_results.append({
                    "varied_feature": variable_feature_name,
                    "value": var_val,
                    "error": str(ve_inner),
                    "predicted_duration": None
                })

        return jsonify({
            "model_version_used": f"model_{model_version_num}",
            "analysis_results": final_results
        })

    except ValueError as ve: # For payload level validation
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Error during sensitivity analysis with model_{model_version_num}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Sensitivity analysis error: {str(e)}"}), 500


# --- Endpoint 3: Suggest Optimal Time ---
@app.route('/suggest/optimal-time/<model_version_num>', methods=['POST'])
def suggest_optimal_time(model_version_num):
    if model_version_num not in loaded_models or loaded_models[model_version_num] is None:
        return jsonify({"error": f"Model 'model_{model_version_num}' not loaded."}), 404
    model_to_use = loaded_models[model_version_num]

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No input payload provided"}), 400

        base_conditions = payload.get('base_conditions')
        target_duration_max = payload.get('target_duration_max')
        target_duration_min = payload.get('target_duration_min', 0) # Default min to 0 if not provided
        hours_to_evaluate = payload.get('hours_to_evaluate')
        minute_of_hour = payload.get('minute_of_hour', 0)

        if not all([base_conditions, hours_to_evaluate]) or target_duration_max is None :
            return jsonify({"error": "Missing 'base_conditions', 'hours_to_evaluate', or 'target_duration_max'."}), 400
        if not isinstance(hours_to_evaluate, list):
            return jsonify({"error": "'hours_to_evaluate' must be a list of integers (0-23)."}), 400
        try:
            target_duration_max = float(target_duration_max)
            target_duration_min = float(target_duration_min)
            minute_of_hour = int(minute_of_hour)
            if not (0 <= minute_of_hour <= 59):
                raise ValueError("minute_of_hour must be between 0 and 59.")
        except ValueError as ve_conv:
            return jsonify({"error": f"Invalid type for target durations or minute: {ve_conv}"}), 400

        suggestions = []
        prediction_rows_hourly = []
        hours_for_spark_processing = [] # Keep track of hours that pass validation

        for hour in hours_to_evaluate:
            if not (0 <= hour <= 23 and isinstance(hour, int)):
                print(f"Skipping invalid hour: {hour}")
                continue

            current_features = base_conditions.copy()
            current_features['Phour'] = float(hour)
            current_features['Pmin'] = float(minute_of_hour)
            # PDweek should be provided in base_conditions. If it needs to be derived from Pmonth/Pday,
            # that logic would go here, potentially requiring Pyear or assuming one.
            # For now, assume PDweek is correctly set in base_conditions.

            try:
                input_tuple, _ = validate_and_prepare_single_input(current_features)
                prediction_rows_hourly.append(input_tuple)
                hours_for_spark_processing.append(hour) # Store the hour corresponding to this row
            except ValueError as ve:
                print(f"Validation error for hour {hour}: {ve}. Skipping this hour.")


        if not prediction_rows_hourly:
            return jsonify({"message": "No valid hours to evaluate after validation.", "suggestions": []}), 200
            
        input_df_hourly = spark.createDataFrame(prediction_rows_hourly, schema=prediction_schema)
        predictions_df_hourly = model_to_use.transform(input_df_hourly)
        predicted_durations_hourly = [row.prediction for row in predictions_df_hourly.select("prediction").collect()]

        for i, hour_processed in enumerate(hours_for_spark_processing):
            predicted_duration = predicted_durations_hourly[i]
            if target_duration_min <= predicted_duration <= target_duration_max:
                suggestions.append({
                    "hour_of_day": hour_processed,
                    "minute_of_hour": minute_of_hour,
                    "predicted_duration": predicted_duration,
                    "within_target_range": True
                })
        
        suggestions.sort(key=lambda x: x['predicted_duration']) # Sort by shortest duration

        return jsonify({
            "model_version_used": f"model_{model_version_num}",
            "target_duration_min": target_duration_min,
            "target_duration_max": target_duration_max,
            "suggestions": suggestions
        })

    except ValueError as ve: # For payload level validation
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Error during optimal time suggestion with model_{model_version_num}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Optimal time suggestion error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)