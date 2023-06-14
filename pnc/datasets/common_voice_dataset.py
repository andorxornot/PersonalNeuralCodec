import os
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset

class CommonVoiceDataset(Dataset):
    def __init__(self, data_root, tsv_file, sample_rate=22050, transform=None):
        """
        Initialize the Common Voice dataset.

        Args:
            data_root (str): Path to the root directory containing the clips folder and TSV files.
            tsv_file (str): Filename of the TSV file containing metadata (e.g., "train.tsv", "test.tsv", "dev.tsv").
            sample_rate (int, optional): Sample rate to use for the audio data.
            transform (callable, optional): Optional transform to apply to the audio data.
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.metadata = pd.read_csv(os.path.join(data_root, tsv_file), delimiter="\t")
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the path to the audio file and load it
        audio_file = os.path.join(self.data_root, "clips", self.metadata.iloc[idx]["path"])
        waveform, sample_rate = librosa.load(audio_file, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        # Apply the optional transform
        if self.transform:
            waveform = self.transform(waveform)

        # Get the corresponding text label
        text = self.metadata.iloc[idx]["sentence"]

        return waveform, text


if __name__ == "__main__":
    data_root = "/path/to/common_voice/data"
    train_tsv = "train.tsv"

    train_dataset = CommonVoiceDataset(data_root, train_tsv)

    # Access a sample from the dataset
    waveform, text = train_dataset[0]
    print("Waveform shape:", waveform.shape)
    print("Text:", text)
