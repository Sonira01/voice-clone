import os
import torch
import torchaudio
import json
from vits import mel_processing

# Load hyperparameters from config
with open("configs/config.json", "r") as f:
    hps = json.load(f)

def get_spec(wav_path):
    wav, _ = torchaudio.load(wav_path)
    wav = wav.mean(0, keepdim=True)  # Convert to mono if needed
    spec = mel_processing.mel_spectrogram(
        wav, hps["data"]
    )
    return spec.squeeze(0)

csv_path = "data/ravdess/metadata.cleaned.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    audio_path = line.strip().split("|", 1)[0].strip()
    if not audio_path.endswith(".wav"):
        continue
    spec_path = audio_path.replace(".wav", ".spec.pt")
    if os.path.exists(spec_path):
        continue
    try:
        spec = get_spec(audio_path)
        torch.save(spec, spec_path)
        print(f"✅ Created: {spec_path}")
    except Exception as e:
        print(f"[❌] Failed on {audio_path}: {e}")
