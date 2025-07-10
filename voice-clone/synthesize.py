import torch
import numpy as np
import soundfile as sf
import librosa
import io

from models import SynthesizerTrn
import utils
from text import text_to_sequence
from resemblyzer import VoiceEncoder, preprocess_wav

# --- Config ---
CONFIG_PATH = "configs/ljs_base.json"
CHECKPOINT_PATH = "fine_tuned.pth"
INPUT_WAV = "input.wav"      # replace with your file
TEXT = "This is my cloned voice speaking."
OUTPUT_WAV = "cloned_voice.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model ---
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

# --- Load and preprocess input voice ---
wav, sr = librosa.load(INPUT_WAV, sr=None)
text_seq = text_to_sequence(TEXT, hps.data.text_cleaners)
if getattr(hps.data, "add_blank", False):
    import commons
    text_seq = commons.intersperse(text_seq, 0)

text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(DEVICE)
text_lengths = torch.LongTensor([text_tensor.size(1)]).to(DEVICE)

g_tensor = None
if hps.model.gin_channels > 0:
    encoder = VoiceEncoder()
    wav_preprocessed = preprocess_wav(wav, source_sr=sr)
    embed = encoder.embed_utterance(wav_preprocessed)
    g_tensor = torch.FloatTensor(embed).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    audio = model.infer(
        text_tensor, text_lengths, sid=None,
        noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0,
        g=g_tensor
    )[0][0, 0].cpu().numpy()

sf.write(OUTPUT_WAV, audio, hps.data.sampling_rate)
print("✅ Cloned voice saved to:", OUTPUT_WAV)
