import numpy as np
import tensorflow as tf
import os

# matching train_kws.py (training)
SAMPLE_RATE = 16000
FFT_LENGTH = 512
NUM_MEL_BINS = 40
LOWER_EDGE_HERTZ = 20.0
UPPER_EDGE_HERTZ = 4000.0

# takes 257 raw bins and combines them mathematically to produce the 40 mel bins that the model actually sees
NUM_FFT_BINS = (FFT_LENGTH // 2) + 1 # so here half is used + its offsett the dc frequency 

def generate_header():
    # using tensorflow as all the math has to be consistent 
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=NUM_FFT_BINS,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_EDGE_HERTZ,
        upper_edge_hertz=UPPER_EDGE_HERTZ
    ).numpy()

    output_dir = "firmware/include"
    output_path = os.path.join(output_dir, "mel_constants.h")
    os.makedirs(output_dir, exist_ok=True)
    
    # writing the file
    with open(output_path, "w") as f:
        f.write("// Generated automatically by gen_mel_header.py\n")
        f.write("// Use this to convert raw FFT output into Mel-Frequency bins\n\n")
        f.write("#ifndef MEL_CONSTANTS_H\n")
        f.write("#define MEL_CONSTANTS_H\n\n")

        f.write(f"#define NUM_FFT_BINS {NUM_FFT_BINS}\n")
        f.write(f"#define NUM_MEL_BINS {NUM_MEL_BINS}\n\n")
        f.write(f"const float mel_filterbank[{NUM_FFT_BINS}][{NUM_MEL_BINS}] = {{\n")
        for i, row in enumerate(linear_to_mel_weight_matrix):
            f.write("    {")
            f.write(", ".join(f"{val:.6f}f" for val in row))
            f.write("},\n")
        f.write("};\n\n")
        f.write("#endif // MEL_CONSTANTS_H\n")
    print("mel_constants.h\ created")

if __name__ == "__main__":
    generate_header()