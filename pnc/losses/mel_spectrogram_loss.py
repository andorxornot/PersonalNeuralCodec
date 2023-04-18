import torch
import torch.nn as nn

class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        # Initialize the mean squared error loss function
        super(MelSpectrogramLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel_spec1, mel_spec2):
        # Compute the mean squared error loss between the two mel spectrograms
        loss = self.mse_loss(mel_spec1, mel_spec2)
        return loss
