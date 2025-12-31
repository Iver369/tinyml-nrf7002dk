import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile 
from wav_to_c import process_audio
import sys

if len(sys.argv) < 2:
    print("Usage: python wav_to_c.py <input_file.wav>")
    sys.exit(1)
input_file = sys.argv[1]

audio_data, fs, raw_all, peak = process_audio(input_file)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
time = np.arange(len(audio_data)) / fs 

ax1.plot(time, audio_data)
ax1.set_title('Waveform')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')

ax2.specgram(audio_data, Fs=fs, NFFT=256, noverlap=128, cmap='plasma')
ax2.set_ylabel("Frequency [Hz]")
ax2.set_xlabel('Time [s]')
ax2.set_title('Spectrogram')
plt.tight_layout()
plt.show()