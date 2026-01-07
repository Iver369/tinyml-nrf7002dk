import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
import sys
from wav_to_c import process_audio

# configuring dsp to match train_kws
SAMPLE_RATE = 16000
FRAME_LENGTH = 512    
FRAME_STEP = 320      
FFT_LENGTH = 512
NUM_MEL_BINS = 40

def compute_log_mel_spectrogram(audio_int16):
    # normalizing float
    audio_float = audio_int16.astype(np.float32) / 32768.0
    audio_tensor = tf.convert_to_tensor(audio_float)

    # slices the audio in little overlapping chunks and performs fast fourier transforms on each chunk
    stft = tf.signal.stft(
        audio_tensor, 
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH
    )
    spectrogram = tf.abs(stft) # returns complex numbers and takes the absolute value by applying the pythagorean theorem to find the magnitude (ignoring the phase)

    # group higher frequencies akin to human ears
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    return log_mel_spectrogram.numpy() # convert to standard python array from tensors so that matplotlib can understand

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/scripts/visualize_dsp.py <input_file.wav>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    print(f"Processing {input_file}...")
    audio_data = process_audio(input_file)
    mel_spec = compute_log_mel_spectrogram(audio_data)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    time = np.arange(len(audio_data)) / SAMPLE_RATE
    ax1.plot(time, audio_data)
    ax1.set_title(f'Raw Waveform (Int16) - {len(audio_data)} samples')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # plotting the mel-spectrogram (what the AI sees)
    im = ax2.imshow(mel_spec.T, aspect='auto', origin='lower', cmap='inferno')
    ax2.set_title(f'Log-Mel Spectrogram (AI Input) - {mel_spec.shape}')
    ax2.set_ylabel(f'Mel Frequency Bins ({NUM_MEL_BINS})')
    ax2.set_xlabel(f'Time Steps ({mel_spec.shape[0]})')
    fig.colorbar(im, ax=ax2, format='%+2.0f dB')

    plt.tight_layout()
    plt.show()