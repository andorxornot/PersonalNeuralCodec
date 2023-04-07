import os
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram


class WavFolderDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Initialize the WavFolderDataset.

        Args:
            data_root (str): Path to the root directory containing the WAV files.
            transform (callable, optional): Optional transform to apply to the audio data.
        """
        self.data_root = data_root
        self.file_list = [f for f in os.listdir(data_root) if f.endswith('.wav')]
        self.transform = transform
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=44100,
            n_fft=400,
            hop_length=None,
            win_length=None,
            n_mels=128
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the path to the audio file and load it
        audio_file = os.path.join(self.data_root, self.file_list[idx])
        waveform, sample_rate = torchaudio.load(audio_file)

        # Compute the mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Apply the optional transform
        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec


if __name__ == "__main__":
    data_root = "/path/to/wav/folder"

    wav_dataset = WavFolderDataset(data_root)

    # Access a sample from the dataset
    mel_spec = wav_dataset[0]
    print("Mel spectrogram shape:", mel_spec.shape)
