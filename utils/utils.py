import os
import time
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim

def unique_timestamp_str():
    """
    Generate a unique timestamp string.

    Returns:
        str: A unique timestamp string in the format YYYY-MM-DD_HH-MM-SS.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())


def wav_to_mel_spectrogram(waveform, sample_rate, n_mels=128):
    """
    Convert a waveform to a mel spectrogram.

    Args:
        waveform (Tensor): Input waveform (channels, time).
        sample_rate (int): Sample rate of the input waveform.
        n_mels (int, optional): Number of mel filters. Default: 128.

    Returns:
        Tensor: Mel spectrogram (channels, n_mels, time).
    """
    mel_spectrogram_transform = MelSpectrogram(sample_rate, n_mels=n_mels)
    mel_spec = mel_spectrogram_transform(waveform)
    return mel_spec


def mel_spectrogram_to_wav(mel_spec, sample_rate, n_iter=32, n_mels=128):
    """
    Convert a mel spectrogram to a waveform using the Griffin-Lim algorithm.

    Args:
        mel_spec (Tensor): Input mel spectrogram (channels, n_mels, time).
        sample_rate (int): Sample rate of the output waveform.
        n_iter (int, optional): Number of iterations for the Griffin-Lim algorithm. Default: 32.
        n_mels (int, optional): Number of mel filters. Default: 128.

    Returns:
        Tensor: Reconstructed waveform (channels, time).
    """
    inverse_mel_scale_transform = InverseMelScale(n_mels=n_mels, sample_rate=sample_rate)
    griffin_lim_transform = GriffinLim(n_iter=n_iter)

    spec = inverse_mel_scale_transform(mel_spec)
    waveform = griffin_lim_transform(spec)
    return waveform


if __name__ == "__main__":
    # Example usage of the helper functions
    print(unique_timestamp_str())
