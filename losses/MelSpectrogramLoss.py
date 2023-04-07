import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram

class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        # Initialize the mean squared error loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, mel_spec1, mel_spec12):
        # Compute the mean squared error loss between the two mel spectrograms
        loss = self.mse_loss(mel_spec1, mel_spec2)
        return loss
