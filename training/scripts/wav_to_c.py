import numpy as np
from scipy.io import wavfile
import sys
import os
# wav files from command line arguments to array of numbers representing the sound
TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH = 16000
C_HEADER_DEFINES = """
#define SAMPLE_RATE 16000
#define NUM_SAMPLES 16000
#define FFT_SIZE 512
#define NUM_FFT_BINS 257     // (512 / 2) + 1
#define NUM_MEL_BINS 40      // target size
#define WINDOW_STEP 320      // 20ms stride
#define SPECTROGRAM_ROWS 49  // (16000 - 512) / 320 + 1
#define SPECTROGRAM_SIZE (SPECTROGRAM_ROWS * NUM_MEL_BINS)
"""

def process_audio(file_path):
    try:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       sample_rate, data = wavfile.read(file_path)
    except ValueError:
        return np.zeros(TARGET_LENGTH, dtype=np.int16)

    if sample_rate!=TARGET_SAMPLE_RATE:
        print(f"Warning: {file_path} is {sample_rate}Hz. Expected 16000Hz")

    if len(data.shape) > 1:
        data = data[:, 0]

    # center the audio, helps if recording has silence at the start / end 
    if len(data) > 0: 
        peak_index = np.argmax(np.abs(data)) 
    else: 
        peak_index = 0
    start_index =peak_index-(TARGET_LENGTH // 2)
    end_index = peak_index+(TARGET_LENGTH // 2)
    
    # handle edge cases for pad / crop
    if start_index < 0:
        end_index -= start_index; start_index = 0
    if end_index > len(data):
        start_index -= (end_index - len(data)); end_index = len(data)
    if start_index < 0: 
        start_index = 0
    
    data_1s = data[start_index:end_index]
    
    # if it is too short, pad with zeroes
    if len(data_1s) < TARGET_LENGTH:
        padding = np.zeros(TARGET_LENGTH - len(data_1s), dtype=np.int16)
        data_1s = np.concatenate((data_1s, padding))
    return data_1s[:TARGET_LENGTH].astype(np.int16)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wav_to_c.py <file1.wav> ...")
        sys.exit(1)

    header_path = 'firmware/include/data.h'
    source_path = 'firmware/src/data.c'
    
    print(f"Generating {header_path} and {source_path}...")

    # generate header
    with open(header_path, 'w') as h:
        h.write('#ifndef DATA_H\n#define DATA_H\n\n#include <stdint.h>\n\n')
        h.write(C_HEADER_DEFINES + '\n')
        for input_file in sys.argv[1:]:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            var_name = f"audio_{base_name.replace(' ', '_').replace('-', '_').lower()}"
            h.write(f'extern const int16_t {var_name}[NUM_SAMPLES];\n')
        h.write('\n#endif // DATA_H\n')

    # data arrays 
    with open(source_path, 'w') as c:
        c.write('#include "data.h"\n\n')
        for input_file in sys.argv[1:]:
            if not os.path.exists(input_file): continue
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            var_name = f"audio_{base_name.replace(' ', '_').replace('-', '_').lower()}"
            data_1s = process_audio(input_file)

            c.write(f'// Generated from {input_file} (Original Volume)\n')
            c.write(f'const int16_t {var_name}[NUM_SAMPLES] __attribute__((aligned(16))) = {{\n    ')
            for i, sample in enumerate(data_1s):
                c.write(f'{sample}, ')
                if (i + 1) % 15 == 0: c.write('\n    ')
            c.write('\n};\n\n')

    print("Done")