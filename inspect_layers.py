import tensorflow as tf
import os

MODEL_PATH = "skin_disease_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("\nLayer Names (Reverse Order):")
    for i, layer in enumerate(reversed(model.layers)):
        print(f"{-i}: {layer.name} ({layer.__class__.__name__})")
        if i > 50: break # Just show the last few layers
        
except Exception as e:
    print(f"Error loading model: {e}")
