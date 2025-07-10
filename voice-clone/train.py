import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from models import SynthesizerTrn
from vits.data_utils import TextAudioLoader, TextAudioCollate

def main():
    hps = utils.get_hparams_from_file("configs/config.json")

    model = SynthesizerTrn(
        len(hps.symbols),
        hps.model.spec_channels,
        hps.model.segment_size // hps.data.hop_length,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        n_speakers=hps.model.n_speakers,
        gin_channels=hps.model.gin_channels,
        use_sdp=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset = TextAudioLoader(hps.data.training_files, hps)
    collate_fn = TextAudioCollate()
    dataloader = DataLoader(
        dataset,
        batch_size=getattr(hps.train, "batch_size", 1),  # fallback if batch_size not defined
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = optim.Adam(model.parameters(), lr=hps.train.learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, hps.train.epochs + 1):
        for i, (text, text_lengths, spec, spec_lengths, wav, wav_lengths) in enumerate(dataloader):
            text, text_lengths = text.to(device), text_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            wav, wav_lengths = wav.to(device), wav_lengths.to(device)

            optimizer.zero_grad()

            # ✅ Safer n_speakers check
            n_speakers = hps.data.n_speakers if hps.data.n_speakers is not None else 0

            if n_speakers > 1:
                sid = ...  # TODO: Extract speaker ID from batch
                y_hat, *_ = model(text, text_lengths, spec, spec_lengths, wav, wav_lengths, sid)
            else:
                y_hat, *_ = model(text, text_lengths, spec, spec_lengths, wav, wav_lengths)

            if y_hat.shape != spec.shape:
                print(f"[❌] Skipping batch: shape mismatch y_hat: {y_hat.shape}, spec: {spec.shape}")
                continue

            loss = criterion(y_hat, spec)
            loss.backward()
            optimizer.step()

            if i % hps.train.log_interval == 0:
                print(f"[✅] Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

    # ✅ Save only once after all epochs
    torch.save({'model': model.state_dict()}, "fine_tuned.pth")
    print("✅ Final model saved as fine_tuned.pth")

if __name__ == "__main__":
    main()
