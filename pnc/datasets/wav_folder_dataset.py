import os
import torch
import librosa
from torch.utils.data import Dataset

class WavFolderDataset(Dataset):
    def __init__(self, data_root, sample_rate=22050, transform=None):
        """
        Initialize the WavFolderDataset.

        Args:
            data_root (str): Path to the root directory containing the WAV files.
            transform (callable, optional): Optional transform to apply to the audio data.
        """
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.file_list = [f for f in os.listdir(data_root) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the path to the audio file and load it
        audio_file = os.path.join(self.data_root, self.file_list[idx])
        waveform, sample_rate = librosa.load(audio_file, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        # Apply the optional transform
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, ""


if __name__ == "__main__":
    data_root = "/path/to/wav/folder"

    wav_dataset = WavFolderDataset(data_root)

    # Access a sample from the dataset
    waveform = wav_dataset[0]
    print("Waveform shape:", waveform.shape)
