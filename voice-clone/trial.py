import os
from pathlib import Path

input_dir = Path("data/ravdess/wavs")  # Change if your path is different
output_dir = Path("data/ravdess/wavs_22050")
output_dir.mkdir(parents=True, exist_ok=True)

for wav_file in input_dir.glob("*.wav"):
    out_file = output_dir / wav_file.name
    cmd = f"ffmpeg -y -i \"{wav_file}\" -ar 22050 \"{out_file}\""
    os.system(cmd)

print("✅ All files resampled to 22050 Hz!")
