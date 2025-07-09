import os
import csv

# Adjust paths
wav_dir = "data/your_dataset/wavs"
metadata_path = "data/your_dataset/metadata.csv"

# Placeholder transcript
default_text = "This is a sample voice for training a voice cloning model."

with open(metadata_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="|")
    for file in os.listdir(wav_dir):
        if file.endswith(".wav"):
            filename = file
            text = default_text
            speaker_id = 0 
            writer.writerow([filename, text, speaker_id])

print(f"Generated metadata.csv with {len(os.listdir(wav_dir))} entries.")
