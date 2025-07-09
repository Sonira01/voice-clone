import os
import librosa
import numpy as np
import csv
from tqdm import tqdm

# === Paths ===
wav_dir = "data/ravdess/wavs"
mel_dir = "data/ravdess/mels"
metadata_path = "data/ravdess/metadata.csv"
default_text = "This is a sample voice for training a voice cloning model."

# === Create Mel Output Dir ===
os.makedirs(mel_dir, exist_ok=True)

# === Preprocessing ===
metadata = []

print("[INFO] Processing RAVDESS files...")
for file in tqdm(os.listdir(wav_dir)):
    if not file.endswith(".wav"):
        continue

    # Full path
    file_path = os.path.join(wav_dir, file)

    # Extract speaker ID (last part of filename before .wav)
    try:
        speaker_id = int(file.split("-")[-1].replace(".wav", "")) - 1  # 24 → 23
    except:
        print(f"[WARN] Skipping malformed filename: {file}")
        continue

    # Load + trim audio
    y, sr = librosa.load(file_path, sr=22050)
    y_trimmed, _ = librosa.effects.trim(y)

    # Convert to mel spectrogram
    mel = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Save Mel as .npy
    mel_name = file.replace(".wav", ".npy")
    mel_path = os.path.join(mel_dir, mel_name)
    np.save(mel_path, mel_db)

    # Add to metadata: relative path | transcript | speaker_id
    metadata.append([f"wavs/{file}", default_text, speaker_id])

# === Write metadata.csv ===
with open(metadata_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    writer.writerows(metadata)

print(f"[✅] Processed {len(metadata)} files")
print(f"[📄] Metadata saved to: {metadata_path}")
print(f"[💾] Mel-spectrograms saved to: {mel_dir}")
