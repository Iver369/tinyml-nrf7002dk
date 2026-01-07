import os
import sys
import numpy as np
import shutil
from pathlib import Path
import tensorflow as tf
from scipy.io import wavfile

MODEL_KERAS_PATH = "training/outputs/model.keras"
TFLITE_FILE = "firmware/src/model.tflite"
C_MODEL_FILE = "firmware/src/model.cpp"  
DATASET_DIR = Path('training/datasets/speech_commands_v0.02').resolve()
SAMPLE_RATE = 16000

print(f"Configuration")
print(f"Model Input: {os.path.abspath(MODEL_KERAS_PATH)}")
print(f"Dataset Dir: {DATASET_DIR}")
print(f"TFLite Output: {os.path.abspath(TFLITE_FILE)}")

def load_audio_file(path):
    try:
        _, data = wavfile.read(path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / 32768.0
        if len(data) < 16000:
            data = np.concatenate([data, np.zeros(16000 - len(data), dtype=np.float32)])
        else:
            data = data[:16000]
        return data
    except Exception:
        return None

def extract_features_for_calibration(audio_data):
    audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    stft = tf.signal.stft(audio_tensor, frame_length=512, frame_step=320, fft_length=512)
    spectrogram = tf.abs(stft)
    
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0
    )
    mel_spectrogram =tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram=tf.math.log(mel_spectrogram + 1e-6)
    return tf.reshape(log_mel_spectrogram, (1, 49, 40, 1))

def representative_dataset():
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found at {DATASET_DIR}")
        raise FileNotFoundError("Dataset missing")

    all_wavs = list(DATASET_DIR.rglob("*.wav"))
    print(f"Found {len(all_wavs)} .wav files in dataset")
    
    if len(all_wavs) == 0:
        print("No .wav files found")
        return

    np.random.shuffle(all_wavs)
    
    count = 0
    for wav_path in all_wavs:
        if count >= 100: break
        
        audio = load_audio_file(wav_path)
        if audio is not None:
            input_tensor = extract_features_for_calibration(audio)
            yield [input_tensor]
            count += 1
            if count % 20 == 0: print(f"Colected {count} samples...")

def hex_to_c_array(data, var_name):
    c_str = ""
    c_str += "#include <model.h>\n\n"
    c_str += f"const unsigned char {var_name}[] __attribute__((aligned(16))) = {{\n"
    for i, val in enumerate(data):
        c_str += f"0x{val:02x}, "
        if (i + 1) % 12 == 0:
            c_str += "\n"
    c_str += "};\n\n"
    c_str += f"const unsigned int {var_name}_len = sizeof({var_name});\n"
    return c_str

def quantize():
    if not os.path.exists(MODEL_KERAS_PATH):
        print(f"Model file missing at {MODEL_KERAS_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_KERAS_PATH)

    TEMP_DIR = "training/outputs/temp_quant_model"
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    model.export(TEMP_DIR) # workaround keras 3 bug, by temporarily exporting 

    converter = tf.lite.TFLiteConverter.from_saved_model(TEMP_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_FILE), exist_ok=True)
    with open(TFLITE_FILE, "wb") as f:
        f.write(tflite_model)
    
    print(f"\nModel written to {TFLITE_FILE}")
    print(f"Size: {len(tflite_model)} bytes")
    c_code = hex_to_c_array(tflite_model, "g_model")
    with open(C_MODEL_FILE, "w") as f:
        f.write(c_code)
    print(f"C++ model updated")

    if os.path.exists(TEMP_DIR): 
        shutil.rmtree(TEMP_DIR)

quantize()