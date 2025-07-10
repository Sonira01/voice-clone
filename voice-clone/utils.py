import json
import torch
import soundfile as sf
import numpy as np
import os

class HParams:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = HParams(v)  # Recursively convert nested dicts
            self.__dict__[k] = v

    def __getattr__(self, name):
        return self.__dict__.get(name, None)  # Avoid KeyError

def get_hparams_from_file(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    return HParams(data)

def load_checkpoint(path, model, _):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

def load_wav_to_torch(full_path):
    data, sampling_rate = sf.read(full_path)
    if len(data.shape) == 2:  # Convert stereo to mono
        data = np.mean(data, axis=1)
    data = torch.FloatTensor(data)
    return data, sampling_rate

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split)[:2] for line in f]
    # Ensure correct audio path
    for i, (wav, text) in enumerate(filepaths_and_text):
        if not os.path.isabs(wav):
            wav = os.path.join('data/ravdess', wav) if not wav.startswith('data/ravdess') else wav
        filepaths_and_text[i][0] = wav
    return filepaths_and_text
