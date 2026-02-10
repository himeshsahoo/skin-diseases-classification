import tensorflow as tf
import os

MODEL_PATH = "skin_disease_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    
    # Also print layer names to be sure
    print("\nLayer Names:")
    for layer in model.layers:
        print(layer.name)
        
except Exception as e:
    print(f"Error loading model: {e}")
