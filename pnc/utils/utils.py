import os
import sys
import time
import torch
import librosa
import random
import numpy as np
import soundfile as sf


def pytorch_worker_info(group=None):
    rank, world_size = None, None
    try:
        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
    except AttributeError:
        pass
    return rank, world_size, None, None  # TODO: the rest of the parameters


def unique_timestamp_str():
    """
    Generate a unique timestamp string.

    Returns:
        str: A unique timestamp string in the format YYYY-MM-DD_HH-MM-SS.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())


def wav_to_mel_spectrogram(waveform, sample_rate, n_mels=80, n_fft=1024, hop_length=256, win_length=None, window_fn="hann"):
    """
    Convert a waveform to a mel spectrogram.

    Args:
        waveform (ndarray): Input waveform (time).
        sample_rate (int): Sample rate of the input waveform.
        n_mels (int, optional): Number of mel filters. Default: 80.
        n_fft (int, optional): Length of the FFT window. Default: 1024.
        hop_length (int, optional): Number of audio samples between adjacent STFT columns. Default: 256.
        win_length (int, optional): Window size. Default: n_fft.
        window_fn (callable, optional): A function to create a window tensor.

    Returns:
        ndarray: Mel spectrogram (n_mels, time).
    """
  
    if win_length is None:
        win_length = n_fft

    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_fn, n_mels=n_mels, power=1)
    return mel_spectrogram


def mel_normalize(mel_spec, min_level_db=-100):
    # Convert MelSpectrogram to dB scale
    mel_db = librosa.power_to_db(mel_spec, ref=1.0)

    # Normalize MelSpectrogram in dB scale with minimum threshold
    mel_norm_db = np.maximum(0, (mel_db - min_level_db) / -min_level_db)
    return mel_norm_db


def mel_denormalize(mel_db, min_level_db=-100):
    # Denormalize MelSpectrogram in dB scale with minimum threshold
    mel_denorm_db = np.maximum(0, mel_db * -min_level_db) + min_level_db

    # Convert dB scale back to power spectrogram
    mel_spec = librosa.db_to_power(mel_denorm_db, ref=1.0)
    return mel_spec


def mel_spectrogram_to_wav(mel_spec, sample_rate, n_mels=80, n_fft=1024, hop_length=256, win_length=None, window_fn="hann"):
    """
    Convert a Mel spectrogram back to a waveform.

    Args:
        mel_spec (ndarray): Mel spectrogram (n_mels, time).
        sample_rate (int): Sample rate of the output waveform.
        n_mels (int, optional): Number of mel filters. Default: 80.
        n_fft (int, optional): Length of the FFT window. Default: 1024.
        hop_length (int, optional): Number of audio samples between adjacent STFT columns. Default: 256.
        win_length (int, optional): Window size. Default: n_fft.

    Returns:
        ndarray: Reconstructed waveform (time).
    """
    if win_length is None:
        win_length = n_fft

    # Inverse Mel scale
    waveform = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_fn, power=1)

    return waveform


class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, waveform):
        start_idx = random.randint(0, waveform.shape[-1] - self.crop_size)
        return waveform[..., start_idx:start_idx + self.crop_size]


if __name__ == "__main__":
    # Example usage of the helper functions
    print(unique_timestamp_str())

    # Test mel spectrogram
    waveform, sample_rate = librosa.load("./data/gettysburg.wav")
    mels = wav_to_mel_spectrogram(waveform, sample_rate=sample_rate)
    mel_db = mel_normalize(mels)

    mels = mel_denormalize(mel_db)
    waveform = mel_spectrogram_to_wav(mels, sample_rate)
    
    sf.write("./tmp.wav", waveform, sample_rate, 'PCM_24')
