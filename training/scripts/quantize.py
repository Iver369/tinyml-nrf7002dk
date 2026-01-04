import tensorflow as tf
import numpy as np
import os

MODEL_DIR = "trained_model" 
TFLITE_FILE = "firmware/src/model.tflite"

def representative_dataset():
    for _ in range(100):
        data = np.random.uniform(0, 1, size=(1, 66, 129, 1)).astype(np.float32)
        yield [data]

def quantize():
    print(f"Loading model from {MODEL_DIR}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    with open(TFLITE_FILE, "wb") as f:
        f.write(tflite_model)
    
    print(f"Quantized model saved to {TFLITE_FILE}")
    print(f"   Size: {len(tflite_model)} bytes")

if __name__ == "__main__":
    quantize()