import numpy as np
from scipy.io import wavfile
import sys

def process_audio(file_path):
    sample_rate, data = wavfile.read(file_path)
    peak_index = np.argmax(np.abs(data)) 
    start_index = peak_index - 8000
    end_index = peak_index + 8000
    if start_index < 0:
        start_index = 0
        end_index = 16000
    if end_index > len(data):
        end_index = len(data)
        start_index = len(data) - 16000

    data_1s = data[start_index:end_index]
    return data_1s, sample_rate, data, peak_index
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wav_to_c.py <input_file.wav>")
        sys.exit(1)
    input_file = sys.argv[1]

    data_1s, sample_rate, data, peak_index = process_audio(input_file)
    print(f"Sample Rate: {sample_rate}")
    print(f"Total Samples: {len(data)}")
    print(f"First 10 Samples: {data[:10]}")
    print(f"Loudest point found at sample: {peak_index}")
    print(f"New total samples for the chip: {len(data_1s)}")
    print(f"First 10 Samples: {data[:10]}")
    print(f"Samples at peak: {data_1s[7995:8005]}")

    with open('firmware/include/data.h', 'w') as f:
        f.write('#include <stdint.h>\n\n')
        f.write('#ifndef DATA_H\n')
        f.write('#define DATA_H\n\n')
        f.write(f'#define SAMPLE_RATE {sample_rate}\n')
        f.write(f'#define NUM_SAMPLES {len(data_1s)}\n\n')
        f.write('const int16_t audio_data[NUM_SAMPLES] = {\n    ')
        for i, sample in enumerate(data_1s):
            f.write(f'{sample}, ')
            if (i + 1) % 10 == 0:
                f.write('\n    ')
        f.write('\n};\n\n')
        f.write('#endif // DATA_H\n')