"""
FastAPI app exposing /predict and /feedback endpoints.
Requires TensorFlow SavedModel directories (arrhythmia_saved_model, binary_saved_model).
Adjust DB credentials in configs/settings.yaml or replace cfg dict below.
"""
from fastapi import FastAPI, HTTPException
from .schemas import InputData, FeedbackData
from . import db_utils
import tensorflow as tf
import numpy as np
from datetime import datetime
import yaml
import os

# Load settings (fallback to minimal defaults)
cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "settings.yaml")
cfg = {}
if os.path.exists(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
db_cfg = cfg.get("database", {"host":"localhost","user":"admin","password":"?BIb9!7xGn4>","database":"ai_prediction_app"})

arr_saved = cfg.get("arr_saved_model_path", "arrhythmia_saved_model")
bin_saved = cfg.get("bin_saved_model_path", "binary_saved_model")

# Load TF models
try:
    arr_net = tf.saved_model.load(arr_saved)
    bin_net = tf.saved_model.load(bin_saved)
except Exception as e:
    raise RuntimeError(f"Failed to load saved models: {e}")

app = FastAPI(title="ECG AI Prediction API")

def run_model(model, X):
    """Run TF SavedModel. Accepts numpy X shaped [batch, seqLen, feat]. Returns numpy logits array."""
    X_tf = tf.constant(X, dtype=tf.float32)
    # attempt to use serving_default signature
    try:
        infer = model.signatures.get("serving_default", None)
        if infer is None:
            # try to call model directly
            out = model(X_tf)
            # out may be dict or tensor
            if isinstance(out, dict):
                return list(out.values())[0].numpy()
            return out.numpy()
        res = infer(X_tf)
        return list(res.values())[0].numpy()
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}")

@app.post("/predict")
def predict(data: InputData):
    model_type = data.model_type.lower()
    if model_type == "arrhythmia":
        model = arr_net
        model_id = 1
    elif model_type == "binary":
        model = bin_net
        model_id = 2
    else:
        raise HTTPException(status_code=400, detail="model_type must be 'arrhythmia' or 'binary'")

    # shape features -> [1, seqLen, 1]
    X = np.array(data.features, dtype=np.float32).reshape(1, -1, 1)
    try:
        output = run_model(model, X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    label_idx = int(np.argmax(output, axis=1)[0])
    confidence = float(np.max(output))
    predicted_label = f"class_{label_idx}"

    # Save to DB
    db = db_utils.get_db_connection(db_cfg)
    cursor = db.cursor()
    input_id, prediction_id = db_utils.save_prediction(cursor, db, data.user_id, model_id, data.features, predicted_label, confidence, data.device_info)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "input_id": input_id,
        "prediction_id": prediction_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/feedback")
def feedback(data: FeedbackData):
    db = db_utils.get_db_connection(db_cfg)
    cursor = db.cursor()
    feedback_id = db_utils.save_feedback(cursor, db, data.prediction_id, data.user_id, data.is_correct, data.corrected_label, data.rating, data.comments)
    return {"feedback_id": feedback_id, "timestamp": datetime.now().isoformat()}
