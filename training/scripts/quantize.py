import tensorflow as tf
import os
import sys
import shutil
from pathlib import Path
from train_kws import load_audio, extract_features

if len(sys.argv) < 2:
    print("Usage: python quantize.py <model_name.keras>")
    sys.exit(1)
model_name = sys.argv[1]

model_path = (f'training/outputs/{model_name}')
model = tf.keras.models.load_model(model_path)

# fix for keras 3 compatability
temp_dir = "temp_tf_saved_model"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
model.export(temp_dir)

def representative_data():
    data_dir = Path('training/datasets/speech_commands_v0.02')
    on_files = list((data_dir / 'on').glob('*.wav'))
    for file_path in on_files[:100]:
        audio = load_audio(file_path)
        spectrogram = extract_features(audio)
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        yield [spectrogram]

converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)


