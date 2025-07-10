# voice-clone/models.py

import torch
import torch.nn as nn
import json
import torch.nn.functional as F

class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.dummy = nn.Linear(10, 2560)  # arbitrary shape for placeholder logic

    def forward(self, text, text_lengths, spec, spec_lengths, wav, wav_lengths):
        B, C, T = spec.shape
        flat_output_size = C * T

        dummy_input = torch.zeros(B, 10, device=spec.device)
        dummy_out = self.dummy(dummy_input)

        dummy_out = F.interpolate(
            dummy_out.unsqueeze(1), size=flat_output_size,
            mode='linear', align_corners=False
        ).squeeze(1)

        return dummy_out.view(B, C, T), None, None, None, None, None

    @torch.no_grad()
    def infer(self, text, text_lengths, sid=None, g=None, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
        B = text.shape[0]
        T = int(22050 * 2.5)  # simulate 2.5 seconds of audio at 22kHz
        audio = torch.randn(B, 1, T).to(text.device)
        return audio, None, None, None


def load_model(config_path, checkpoint_path, device='cpu'):
    with open(config_path, 'r') as f:
        hps = json.load(f)

    model = SynthesizerTrn(
        n_vocab=hps["n_vocab"],
        spec_channels=hps["model"]["spec_channels"],
        segment_size=hps["model"]["segment_size"],
        inter_channels=hps["model"]["inter_channels"],
        hidden_channels=hps["model"]["hidden_channels"],
        filter_channels=hps["model"]["filter_channels"],
        n_heads=hps["model"]["n_heads"],
        n_layers=hps["model"]["n_layers"],
        kernel_size=hps["model"]["kernel_size"],
        p_dropout=hps["model"]["p_dropout"],
        resblock=hps["model"]["resblock"],
        resblock_kernel_sizes=hps["model"]["resblock_kernel_sizes"],
        resblock_dilation_sizes=hps["model"]["resblock_dilation_sizes"],
        upsample_rates=hps["model"]["upsample_rates"],
        upsample_initial_channel=hps["model"]["upsample_initial_channel"],
        upsample_kernel_sizes=hps["model"]["upsample_kernel_sizes"],
        n_speakers=hps["model"].get("n_speakers", 0),
        gin_channels=hps["model"].get("gin_channels", 0),
        use_sdp=hps["model"].get("use_sdp", True)
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    model.eval()
    return model
