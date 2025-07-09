import streamlit as st
import torch
import numpy as np
import soundfile as sf
import os

from .models import SynthesizerTrn
import utils
from text import text_to_sequence

CONFIG_PATH = "configs/ljs_base.json"  # or your config
CHECKPOINT_PATH = "models/my_voice_clone.pth"  # your trained model
DEVICE = "cpu"  # or "cuda" if available

@st.cache_resource
def load_model():
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=getattr(hps.data, "n_speakers", 0),
        **hps.model
    ).to(DEVICE)
    _ = net_g.eval()
    utils.load_checkpoint(CHECKPOINT_PATH, net_g, None)
    return net_g, hps

net_g, hps = load_model()

st.title("Voice Cloning Demo (VITS)")

text = st.text_area("Enter text to synthesize:", "Hello, this is a cloned voice using VITS!")

sid = None
if getattr(hps.data, "n_speakers", 0) > 1:
    sid = st.number_input("Speaker ID (integer)", min_value=0, max_value=hps.data.n_speakers-1, value=0)

if st.button("Synthesize"):
    with st.spinner("Synthesizing..."):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if getattr(hps.data, "add_blank", False):
            import commons
            text_norm = commons.intersperse(text_norm, 0)
        text_tensor = torch.LongTensor(text_norm).unsqueeze(0).to(DEVICE)
        text_lengths = torch.LongTensor([text_tensor.size(1)]).to(DEVICE)
        with torch.no_grad():
            if sid is not None:
                sid_tensor = torch.LongTensor([sid]).to(DEVICE)
                audio = net_g.infer(text_tensor, text_lengths, sid=sid_tensor, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][0,0].cpu().numpy()
            else:
                audio = net_g.infer(text_tensor, text_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][0,0].cpu().numpy()
        sf.write("output.wav", audio, hps.data.sampling_rate)
        st.audio("output.wav", format="audio/wav")
        st.success("Done!")
