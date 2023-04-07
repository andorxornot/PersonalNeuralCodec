import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class CommonVoiceDataset(Dataset):
    def __init__(self, data_root, tsv_file, transform=None):
        """
        Initialize the Common Voice dataset.

        Args:
            data_root (str): Path to the root directory containing the clips folder and TSV files.
            tsv_file (str): Filename of the TSV file containing metadata (e.g., "train.tsv", "test.tsv", "dev.tsv").
            transform (callable, optional): Optional transform to apply to the audio data.
        """
        self.data_root = data_root
        self.metadata = pd.read_csv(os.path.join(data_root, tsv_file), delimiter="\t")
        self.transform = transform
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=44100,
            n_fft=400,
            hop_length=None,
            win_length=None,
            n_mels=128
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the path to the audio file and load it
        audio_file = os.path.join(self.data_root, "clips", self.metadata.iloc[idx]["path"])
        waveform, sample_rate = torchaudio.load(audio_file)

        # Get the corresponding text label
        text = self.metadata.iloc[idx]["sentence"]

        # Compute the mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Apply the optional transform
        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, text


if __name__ == "__main__":
    data_root = "/path/to/common_voice/data"
    train_tsv = "train.tsv"

    train_dataset = CommonVoiceDataset(data_root, train_tsv)

    # Access a sample from the dataset
    mel_spec, text = train_dataset[0]
    print("Mel spectrogram shape:", mel_spec.shape)
    print("Text:", text)
