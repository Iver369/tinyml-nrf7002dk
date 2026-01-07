import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.io import wavfile
import warnings

DATASET_PATH = Path('training/datasets/speech_commands_v0.02')
TARGET_WORD = 'on'

SAMPLE_RATE = 16000
INPUT_SHAPE = (49, 40, 1) # lowering to (49, 40) made it much faster
BATCH_SIZE = 64
EPOCHS= 30 

warnings.filterwarnings("ignore", category=UserWarning)

def load_audio(item):
    if isinstance(item, (Path, str)):
        try:
            _, data = wavfile.read(item)
        except Exception:
            return np.zeros(16000, dtype=np.float32)
    else: 
        data = item

    if data.dtype != np.float32:
        data = data.astype(np.float32) / 32768.0

    if data.ndim > 1: # check if the audio is stereo
        data = data[:, 0] # slicing to only keep one channel (mono)

    if len(data) < 16000: # padding and cropping
        padding = 16000 - len(data)
        data = np.concatenate([data, np.zeros(padding, dtype=np.float32)])
    else:
        data = data[:16000]
    return data

def extract_features(audio_array):
    audio_tensor = tf.convert_to_tensor(audio_array, dtype=tf.float32) #numpy array into tensor

    stft = tf.signal.stft(
        audio_tensor, 
        frame_length=512,  
        frame_step=320,    
        fft_length=512
    )
    spectrogram = tf.abs(stft)
    num_spectrogram_bins = stft.shape[-1]
    # mathematical matrix that mimics human ears
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=4000.0
    )
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1) # matrix multiplication, applies the filter to spectrogram
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])) 
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    return tf.expand_dims(log_mel_spectrogram, axis=-1) # adds 1 dimension here as cnn expects the 3 values


def create_dataset(items, labels, is_training=True): # using this for one at a time to save ram
    def generator():
        for i in range(len(items)):
            audio = load_audio(items[i])
            
            if is_training and random.random() < 0.1: # change a little to make model more robust
                shift = random.randint(-1000,1000)
                audio = np.roll(audio, shift)
                
            yield extract_features(audio), labels[i]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    return dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    print(f"TRAINING START: TARGET '{TARGET_WORD}' = LABEL 2")
    positives = list((DATASET_PATH / TARGET_WORD).glob('*.wav'))
    print(f"Targets: {len(positives)}")
    backgrounds = []
    bg_folder = DATASET_PATH / '_background_noise_'
    for file in bg_folder.glob('*.wav'):
        try:
            _, audio = wavfile.read(file)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / 32768.0
            
            for i in range(0, len(audio) - 16000, 16000): # as background sounds are 60s we have to split it up to make 1 second sounds
                backgrounds.append(audio[i:i+16000])
        except: pass
    backgrounds = backgrounds * 10 # ensures balanced training dataset between unknown, background, and on
    print(f"Background Chunks: {len(backgrounds)}")
    hard_negatives = ['cat', 'dog', 'off', 'one', 'no', 'go', 'up', 'down', 'left', 'right', 'stop'] # one, no, go, off
    
    unknown_files = []
    for folder in [d for d in DATASET_PATH.iterdir() if d.is_dir()]:
        if folder.name == TARGET_WORD or folder.name == '_background_noise_': 
            continue
        files = list(folder.glob('*.wav'))
        if folder.name in hard_negatives:
            unknown_files.extend(files) # training for differences between similar words / opposites 
        else:
            unknown_files.extend(random.sample(files, min(len(files), 200))) # training on more generalized and easier to differentiate sounds

    if len(unknown_files) > 40000: # keeps it capped for dataset balance to avoid model guessing unknown for everything
        unknown_files = random.sample(unknown_files, 40000)
    print(f"Unknowns: {len(unknown_files)}")

    all_items = [] # categorizing by assigning integers
    for b in backgrounds: all_items.append((b, 0))
    for u in unknown_files: all_items.append((u, 1))
    for p in positives: all_items.append((p, 2))

    random.shuffle(all_items) # forcing to decide on sound rather than order
    split_idx = int(len(all_items) * 0.8)
    train_list = all_items[:split_idx] # 80% of data goes to training the model
    val_list = all_items[split_idx:] # 20% of the data remains for validation during training

    print(f"Training: {len(train_list)} | Validation: {len(val_list)}")
    train_items, train_labels = zip(*train_list)
    val_items, val_labels= zip(*val_list)
    train_ds = create_dataset(train_items, train_labels, is_training=True)
    val_ds = create_dataset(val_items, val_labels, is_training=False)
    class_weights = {0: 1.0, 1: 1.0, 2: 5.0} # penalty for not getting on, avoids model to guess for example unknown for everything

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=INPUT_SHAPE),
        tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'), # instead of conv2d for less math/memory usage
        tf.keras.layers.Conv2D(32, (1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2), # for avoiding overfitting 
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1) #if the model stops improving, lowers its rate of learning
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weights, callbacks=[reduce_lr])
    model.save('training/outputs/model.keras')
    print("Model Saved")