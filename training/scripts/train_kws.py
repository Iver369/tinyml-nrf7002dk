from pathlib import Path
import random
import os
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import tensorflow as tf

#data preprocessing pipeline
data_dir = Path('training/datasets/speech_commands_v0.02')
all_data = []
all_labels = []
sampling_rate = 16000
stride = 2000
Categories = [
    'background',
    'words',
    'on'
]

on_files = list((data_dir / 'on').glob('*.wav'))
words_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name not in ['on', '_background_noise_']]
words_files = []
for folder in words_folders:
    files = list(folder.glob('*.wav'))
    words_files.extend(random.sample(files, min(len(files), 100)))
background_files = list((data_dir / '_background_noise_').glob('*.wav'))
background_samples = []
for files in background_files:
    sr, data = wavfile.read(files)
    if data.ndim > 1:
        data = data[:, 0]
    for i in range (0, len(data)-16000, 2000):
        chunk = data[i:i+16000]
        background_samples.append(chunk)

all_data.extend(background_samples)
all_labels.extend([0]*len(background_samples))

all_data.extend(words_files)
all_labels.extend([1]*len(words_files))

all_data.extend(on_files)
all_labels.extend([2]*len(on_files))

print(f"Total Samples: {len(all_data)}")
print(f"Balance: Background({all_labels.count(0)}) | Unknown({all_labels.count(1)}) | ON({all_labels.count(2)})")
print(f"Target Samples (ON): {len(on_files)}")
print(f"Unkown Samples (Mixed Words): {len(words_files)}")

#data normalization
def load_audio(item):
    if isinstance(item, Path):
        sr, data = wavfile.read(item)
    else: 
        data = item
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32768.0

    if len(data) < 16000:
        padding = np.zeros(16000 - len(data), dtype=np.float32)
        data = np.concatenate([data, padding])
    else:
        data = data[:16000]
        
    return data

def extract_features(audio_array):
    audio_tensor = tf.convert_to_tensor(audio_array, dtype=tf.float32)
    stft = tf.signal.stft(audio_tensor, frame_length=256, frame_step=128)
    spectrogram = tf.abs(stft)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram

def operation():
    c = list(zip(all_data, all_labels))
    random.shuffle(c)
    data_shuffled, labels_shuffled = zip(*c)
    split_index = int(len(data_shuffled) * 0.8)

    train_data = data_shuffled[:split_index]
    train_labels = labels_shuffled[:split_index]

    val_data = data_shuffled[split_index:]
    val_labels = labels_shuffled[split_index:]

    return train_data, train_labels, val_data, val_labels


def data_generator(data_list, label_list):
    for i in range(len(data_list)):
        audio = load_audio(data_list[i])
        spectrogram = extract_features(audio)
        yield spectrogram, label_list[i]

train_data, train_labels, val_data, val_labels = operation()
train_dataset = tf.data.Dataset.from_generator(
lambda: data_generator(train_data, train_labels), 
output_signature=(tf.TensorSpec(shape=(124, 129, 1), dtype=tf.float32), 
tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
train_dataset = train_dataset.cache().batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
lambda: data_generator(val_data, val_labels), 
output_signature=(tf.TensorSpec(shape=(124, 129, 1), dtype=tf.float32), 
tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
val_dataset = val_dataset.cache().batch(32).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(124,129,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(3, activation='softmax')
]) 

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=0.00001,
    verbose=1
)

model.fit(
train_dataset, 
validation_data=val_dataset, 
callbacks=[reduce_lr],
epochs=50
)

model.save('training/outputs/model.keras')
