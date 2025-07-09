import os, librosa, numpy as np
from tqdm import tqdm

input_dir = "data/ravdess/wavs"
output_dir = "data/ravdess/mels"
os.makedirs(output_dir, exist_ok=True)

for file in tqdm(os.listdir(input_dir)):
    if file.endswith(".wav"):
        y, sr = librosa.load(os.path.join(input_dir, file), sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        np.save(os.path.join(output_dir, file.replace(".wav", ".npy")), mel_db)
