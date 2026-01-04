import numpy as np
from scipy.io import wavfile
import sys
import os

TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
TARGET_PEAK = 28000  

def normalize_audio(data):
    if len(data) == 0: return data
    
    max_val = np.max(np.abs(data))
    
    if max_val == 0: 
        print("Silence")
        return data 
    

    gain = TARGET_PEAK / max_val
    if max_val < 100:
        print(f"Signal is too weak. (Peak: {max_val})")
    print(f"Normalizing the audio with boosting by {gain:.2f}x")
    return (data * gain).astype(np.int16)

def process_audio(file_path):
    try:
        sample_rate, data = wavfile.read(file_path)
    except ValueError:
        print(f"Could not read {file_path}")
        return np.zeros(TARGET_LENGTH, dtype=np.int16)

    if sample_rate != TARGET_SAMPLE_RATE:
        print(f"{file_path} is {sample_rate}Hz (Expected {TARGET_SAMPLE_RATE}Hz)")

    if len(data) > 0:
        peak_index = np.argmax(np.abs(data)) 
    else:
        peak_index = 0
    
    start_index = peak_index - (TARGET_LENGTH // 2)
    end_index = peak_index + (TARGET_LENGTH // 2)
    if start_index < 0:
        start_index = 0
        end_index = TARGET_LENGTH
    if end_index > len(data):
        end_index = len(data)
        start_index = len(data) - TARGET_LENGTH
        if start_index < 0: start_index = 0 
    data_1s = data[start_index:end_index]

    if len(data_1s) < TARGET_LENGTH:
        padding = np.zeros(TARGET_LENGTH - len(data_1s), dtype=np.int16)
        data_1s = np.concatenate((data_1s, padding))

    data_1s = normalize_audio(data_1s)

    return data_1s

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wav_to_c.py <file1.wav> ...")
        sys.exit(1)

    output_path = 'firmware/include/data.h'
    print(f"Generating {output_path}...")

    with open(output_path, 'w') as f:
        f.write('#include <stdint.h>\n\n')
        f.write('#ifndef DATA_H\n')
        f.write('#define DATA_H\n\n')
        f.write(f'#define SAMPLE_RATE {TARGET_SAMPLE_RATE}\n')
        f.write(f'#define NUM_SAMPLES {TARGET_LENGTH}\n\n')

        for input_file in sys.argv[1:]:
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}")
                continue
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            clean_name = base_name.replace(" ", "_").replace("-", "_").lower()
            var_name = f"audio_{clean_name}"

            print(f"Processing {input_file} -> {var_name}")
            data_1s = process_audio(input_file)

            f.write(f'// Generated from {input_file}\n')
            f.write(f'const int16_t {var_name}[NUM_SAMPLES] __attribute__((aligned(16))) = {{\n    ')
            for i, sample in enumerate(data_1s):
                f.write(f'{sample}, ')
                if (i + 1) % 15 == 0:
                    f.write('\n    ')
            f.write('\n};\n\n')

        f.write('#endif // DATA_H\n')
        print("data.h Updated")