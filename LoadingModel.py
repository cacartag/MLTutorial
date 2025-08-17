import os
from datetime import datetime
import json
import joblib


def load_model_artifacts(model_dir):
    """Load all model artifacts"""

    # Load model
    model = joblib.load(f"{model_dir}/model.pkl")

    #Load encoders
    encoders = joblib.load(f"{model_dir}/encoders.pkl")

    # Load metadata
    with open(f"{model_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)

    print(f"Loaded model: {metadata['model_name']}(v{metadata['timestamp']})")

    return model, encoders, metadata


# Test loading
loaded_model, loaded_encoders, loaded_metadata = load_model_artifacts(model_dir="models/titanic_survival_v20250817_122456")