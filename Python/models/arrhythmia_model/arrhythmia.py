# convert_to_tflite.py
# ====================
# Loads MATLAB-exported model and converts to TFLite

import tensorflow as tf
import arrhythmia_model.model as matlab_model   # MATLAB exported model
from arrhythmia_model import loadWeights
import os

# -------------------------------------------------------------------
# Step 1: Create the model architecture
# -------------------------------------------------------------------
model = matlab_model.create_model()

# -------------------------------------------------------------------
# Step 2: Load MATLAB-exported weights (from weights.h5)
# -------------------------------------------------------------------
weights_path = "weights.h5"
loadWeights(model, filename=weights_path, debug=False)

print("✅ Model and weights loaded successfully.")

# -------------------------------------------------------------------
# Step 3: Save to Keras .h5 (optional)
# -------------------------------------------------------------------
model.save("arrhythmia_model.h5")
print("💾 Saved Keras model: arrhythmia_model.h5")

# -------------------------------------------------------------------
# Step 4: Save as TensorFlow SavedModel (required for TFLite)
# -------------------------------------------------------------------
tf.saved_model.save(model, "arrhythmia_saved_model")
print("💾 Saved TensorFlow SavedModel: arrhythmia_saved_model/")

# -------------------------------------------------------------------
# Step 5: Convert to TensorFlow Lite
# -------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_saved_model("arrhythmia_saved_model")
# Optional optimizations for mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("arrhythmia_model.tflite", "wb") as f:
    f.write(tflite_model)

print("🎉 Exported to arrhythmia_model.tflite")
