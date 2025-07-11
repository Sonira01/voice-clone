import os
import torch
import numpy as np
import soundfile as sf
import librosa
import streamlit as st
from models import SynthesizerTrn
import utils
from text import text_to_sequence
from resemblyzer import VoiceEncoder, preprocess_wav
import pyttsx3
from tempfile import NamedTemporaryFile

# --- Config ---
CONFIG_PATH = "configs/ljs_base.json"
CHECKPOINT_PATH = "fine_tuned.pth"
OUTPUT_WAV = "cloned_voice.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Preprocess audio ---
def preprocess_audio(input_path, output_path="input_clean.wav", target_sr=16000):
    wav, sr = librosa.load(input_path, sr=None)
    wav = librosa.to_mono(wav)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = wav / np.max(np.abs(wav))
    sf.write(output_path, wav, target_sr)
    return output_path, target_sr

# --- VITS Voice Cloning ---
def synthesize_with_vits(text, input_wav_path=None, use_sid=False, speaker_id=0):
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    if isinstance(hps.symbols, str):
        hps.symbols = list(hps.symbols)

    model = SynthesizerTrn(
        n_vocab=len(hps.symbols),
        spec_channels=hps.model.spec_channels,
        segment_size=hps.model.segment_size or 8192,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_speakers=hps.model.n_speakers,
        gin_channels=hps.model.gin_channels,
        use_sdp=getattr(hps.model, "use_sdp", True)
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Text to tensor
    text_seq = text_to_sequence(text, hps.data.text_cleaners)
    if getattr(hps.data, "add_blank", False):
        import commons
        text_seq = commons.intersperse(text_seq, 0)
    text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(DEVICE)
    text_lengths = torch.LongTensor([text_tensor.size(1)]).to(DEVICE)

    sid_tensor = None
    g_tensor = None

    if use_sid:
        sid_tensor = torch.LongTensor([speaker_id]).to(DEVICE)
    elif input_wav_path:
        cleaned_wav_path, sr = preprocess_audio(input_wav_path)
        wav, _ = librosa.load(cleaned_wav_path, sr=sr)
        encoder = VoiceEncoder()
        wav_preprocessed = preprocess_wav(wav, source_sr=sr)
        embed = encoder.embed_utterance(wav_preprocessed)
        g_tensor = torch.FloatTensor(embed).unsqueeze(0).to(DEVICE)

    dynamic_length_scale = max(0.8, len(text) / 20.0)
    with torch.no_grad():
        audio = model.infer(
            text_tensor,
            text_lengths,
            sid=sid_tensor,
            g=g_tensor,
            noise_scale=0.4,
            noise_scale_w=0.6,
            length_scale=dynamic_length_scale
        )[0][0, 0].cpu().numpy()

    sf.write(OUTPUT_WAV, audio, hps.data.sampling_rate)
    return OUTPUT_WAV

# --- Fallback TTS ---
def synthesize_with_tts(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, OUTPUT_WAV)
    engine.runAndWait()
    return OUTPUT_WAV

# --- Streamlit UI ---
st.title("🧬 Voice Cloning with VITS")
text = st.text_area("Enter text to synthesize", "This is my cloned voice speaking.")
uploaded_file = st.file_uploader("Upload reference voice (optional)", type=["wav"])
use_sid = st.checkbox("Use Speaker ID instead of embedding", value=False)
speaker_id = st.number_input("Speaker ID (0–23)", min_value=0, max_value=23, value=0, step=1)

if st.button("🎤 Generate Voice"):
    try:
        temp_path = None
        if uploaded_file:
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

        if use_sid or temp_path:
            result_path = synthesize_with_vits(text, input_wav_path=temp_path, use_sid=use_sid, speaker_id=speaker_id)
        else:
            result_path = synthesize_with_tts(text)

        audio_bytes = open(result_path, "rb").read()
        st.audio(audio_bytes, format="audio/wav")
        st.download_button("⬇️ Download Audio", audio_bytes, file_name="cloned_voice.wav")

    except Exception as e:
        st.error(f"❌ Error: {e}")
