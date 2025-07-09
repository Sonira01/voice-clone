import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import soundfile as sf
from models import load_model
from text import text_to_sequence
import utils

# Load config
hps = utils.get_hparams_from_file("configs/config.json")

# Load model
net_g = load_model(
    config_path="configs/config.json",
    checkpoint_path="models/your_finetuned.pth",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Prepare input text
text = "Hello, I am your AI clone."
sequence = text_to_sequence(text, hps.data.text_cleaners)
sequence = torch.LongTensor(sequence).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# Inference
with torch.no_grad():
    audio = net_g.infer(sequence, torch.LongTensor([sequence.size(1)]).to(sequence.device))[0][0, 0].cpu().numpy()

# Save audio
sf.write("output.wav", audio, hps.data.sampling_rate)
