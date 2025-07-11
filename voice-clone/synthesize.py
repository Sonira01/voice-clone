import os
import torch
import numpy as np
import soundfile as sf
import librosa
from models import SynthesizerTrn
import utils
from text import text_to_sequence
from resemblyzer import VoiceEncoder, preprocess_wav
import pyttsx3

# --- Config ---
CONFIG_PATH = "configs/ljs_base.json"
CHECKPOINT_PATH = "fine_tuned.pth"
TEXT = input("Enter text to synthesize: ") or "This is my cloned voice speaking."
INPUT_WAV = "input.wav"
OUTPUT_WAV = "cloned_voice.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Preprocess audio for clean embedding ---
def preprocess_audio(input_path, output_path="input_clean.wav", target_sr=16000):
    wav, sr = librosa.load(input_path, sr=None)
    wav = librosa.to_mono(wav)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = wav / np.max(np.abs(wav))  # normalize
    sf.write(output_path, wav, target_sr)
    return output_path, target_sr

# --- Clone Voice with VITS ---
def synthesize_with_vits():
    print("🔊 Cloning voice with VITS...")
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    if isinstance(hps.symbols, str):
        hps.symbols = list(hps.symbols)

    # Build model
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

    # --- Preprocess text ---
    text_seq = text_to_sequence(TEXT, hps.data.text_cleaners)
    if getattr(hps.data, "add_blank", False):
        import commons
        text_seq = commons.intersperse(text_seq, 0)
    
    if len(text_seq) == 0:
        raise ValueError("Text sequence is empty after cleaning. Check text_cleaners in config.")

    text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(DEVICE)
    text_lengths = torch.LongTensor([text_tensor.size(1)]).to(DEVICE)

    # --- Preprocess input.wav and get speaker embedding ---
    cleaned_wav_path, sr = preprocess_audio(INPUT_WAV)
    wav, _ = librosa.load(cleaned_wav_path, sr=sr)
    encoder = VoiceEncoder()
    wav_preprocessed = preprocess_wav(wav, source_sr=sr)
    embed = encoder.embed_utterance(wav_preprocessed)
    g_tensor = torch.FloatTensor(embed).unsqueeze(0).to(DEVICE)

    # --- Inference ---
    dynamic_length_scale = max(0.7, len(TEXT) / 20.0)  # Adjust length based on text
    with torch.no_grad():
        audio = model.infer(
            text_tensor, text_lengths, sid=None,
            noise_scale=0.4, noise_scale_w=0.6, length_scale=dynamic_length_scale,
            g=g_tensor
        )[0][0, 0].cpu().numpy()

    sf.write(OUTPUT_WAV, audio, hps.data.sampling_rate)
    print("✅ Cloned voice saved to:", OUTPUT_WAV)

# --- Fallback to TTS if input.wav is missing or fails ---
def synthesize_with_tts():
    print("🗣️ No input.wav found or error occurred. Using fallback TTS...")
    engine = pyttsx3.init()
    engine.save_to_file(TEXT, OUTPUT_WAV)
    engine.runAndWait()
    print("✅ Fallback TTS saved to:", OUTPUT_WAV)

# --- Run Logic ---
if os.path.exists(INPUT_WAV):
    try:
        synthesize_with_vits()
    except Exception as e:
        print("❌ Voice cloning failed. Reason:", str(e))
        synthesize_with_tts()
else:
    synthesize_with_tts()
