import torch
import torch.nn as nn

class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4, 8]):
        """
        Initialize the MultiScaleMelSpectrogramLoss class.

        Args:
            scales (list, optional): List of scales to downsample the mel spectrograms. Default: [1, 2, 4, 8].
        """
        super(MultiScaleMelSpectrogramLoss, self).__init__()
        self.scales = scales
        self.mse_loss = nn.MSELoss()

    def forward(self, mel_spec1, mel_spec2):
        """
        Calculate the multiscale loss between two mel spectrograms.

        Args:
            mel_spec1 (Tensor): The first mel spectrogram (batch_size, channels, freq, time).
            mel_spec2 (Tensor): The second mel spectrogram (batch_size, channels, freq, time).

        Returns:
            Tensor: The multiscale loss between the two mel spectrograms.
        """
        loss = 0

        for scale in self.scales:
            if scale == 1:
                # If scale is 1, no downsampling is needed
                downsampled_mel_spec1 = mel_spec1
                downsampled_mel_spec2 = mel_spec2
            else:
                downsampled_mel_spec1 = F.avg_pool2d(mel_spec1, kernel_size=scale, stride=scale)
                downsampled_mel_spec2 = F.avg_pool2d(mel_spec2, kernel_size=scale, stride=scale)

            scale_loss = self.mse_loss(downsampled_mel_spec1, downsampled_mel_spec2)
            loss += scale_loss

        return loss
