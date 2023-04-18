import os
import sys
import argparse

# Add paths to import from external submodules and internal modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'audiolm-pytorch'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'encodec'))

# Import necessary libraries and modules
from torch.utils.data import DataLoader
from pnc.datasets.wav_folder_dataset import WavFolderDataset
from pnc.models.vqvae_model import VQVAE
from pnc.losses import MelSpectrogramLoss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment 1 Evaluation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="./logs/experiment1", help="Output directory")

    return parser.parse_args()

def evaluate(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

     # Audio settings
    sample_rate = 22050
    n_mels = 80

    # Load dataset
    transform = RandomCrop(2*4096-256)
    test_dataset = WavFolderDataset("./data", sample_rate=sample_rate, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss, and load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(in_channels=1, out_channels=1, hidden_channels=128, num_embeddings=1024, embedding_dim=8).to(device)
    criterion = MelSpectrogramLoss().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for waveform, text in test_dataloader:
            # Get normalized mel spectogram
            mels = wav_to_mel_spectrogram(waveform.numpy(), sample_rate=sample_rate, n_mels = n_mels)
            mel_db = mel_normalize(mels)
            data = torch.from_numpy(mel_db).to(device)

            output, _, _ = model(data)
            loss = criterion(output, data)
            total_loss += loss.item() * len(data)
            total_samples += len(data)

    avg_loss = total_loss / total_samples
    print(f"Average loss: {avg_loss}")

    # Save evaluation results
    result_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Average loss: {avg_loss}\n")
    print(f"Evaluation results saved at {result_file}")

if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
