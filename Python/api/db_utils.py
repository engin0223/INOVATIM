"""
Simple MySQL helper functions. Adjust connection parameters in configs/settings.yaml or when calling.
"""
import mysql.connector
from datetime import datetime
import json

def get_db_connection(cfg):
    return mysql.connector.connect(
        host=cfg.get("host", "localhost"),
        user=cfg.get("user", "root"),
        password=cfg.get("password", ""),
        database=cfg.get("database", "")
    )

def ensure_user_exists(cursor, user_id):
    cursor.execute("SELECT 1 FROM users WHERE user_id = %s", (user_id,))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO users (user_id, timestamp) VALUES (%s, %s)", (user_id, datetime.now()))

def ensure_model_exists(cursor, model_id, model_name):
    cursor.execute("SELECT 1 FROM models WHERE model_id = %s", (model_id,))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO models (model_id, model_name, deployed_at) VALUES (%s, %s, %s)", (model_id, model_name, datetime.now()))

def save_prediction(cursor, db, user_id, model_id, input_payload, predicted_label, confidence, device_info="API-Client"):
    ensure_user_exists(cursor, user_id)
    cursor.execute(
        """INSERT INTO input_data (user_id, input_type, input_payload, device_info, timestamp)
           VALUES (%s, %s, %s, %s, %s)""",
        (user_id, "sensor", json.dumps(input_payload), device_info, datetime.now())
    )
    input_id = cursor.lastrowid
    cursor.execute(
        """INSERT INTO predictions (input_id, model_id, predicted_label, confidence, latency_ms, timestamp)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (input_id, model_id, predicted_label, confidence, 10, datetime.now())
    )
    prediction_id = cursor.lastrowid
    cursor.execute(
        """INSERT INTO audit_logs (event_type, user_id, details, timestamp) VALUES (%s, %s, %s, %s)""",
        ("response", user_id, json.dumps({"prediction_id": prediction_id, "model_id": model_id, "predicted_label": predicted_label, "confidence": confidence}), datetime.now())
    )
    db.commit()
    return input_id, prediction_id

def save_feedback(cursor, db, prediction_id, user_id, is_correct, corrected_label=None, rating=None, comments=None):
    ensure_user_exists(cursor, user_id)
    cursor.execute(
        """INSERT INTO feedback (prediction_id, user_id, is_correct, corrected_label, rating, comments, timestamp)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (prediction_id, user_id, int(is_correct), corrected_label, rating, comments, datetime.now())
    )
    feedback_id = cursor.lastrowid
    cursor.execute(
        """INSERT INTO audit_logs (event_type, user_id, details, timestamp) VALUES (%s, %s, %s, %s)""",
        ("FEEDBACK", user_id, json.dumps({"feedback_id": feedback_id, "prediction_id": prediction_id, "is_correct": is_correct, "corrected_label": corrected_label, "rating": rating, "comments": comments}), datetime.now())
    )
    db.commit()
    return feedback_id
