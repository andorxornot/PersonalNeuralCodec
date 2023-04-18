import os
import sys
import argparse

# Add paths to import from external submodules and internal modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'audiolm-pytorch'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'encodec'))

# Import necessary libraries and modules
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter

from pnc.datasets.wav_folder_dataset import WavFolderDataset
from pnc.models.vqvae_model import VQVAE
from pnc.losses.multi_scale_mel_spectrogram_loss import MultiScaleMelSpectrogramLoss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment 1 Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./logs/experiment1", help="Output directory")

    return parser.parse_args()

def train(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, unique_timestamp_str()))

     # Audio settings
    sample_rate = 22050
    n_mels = 80

    # Load dataset
    transform = RandomCrop(2*4096-256)
    train_dataset = WavFolderDataset("./data", sample_rate=sample_rate, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(in_channels=1, out_channels=1, hidden_channels=128, num_embeddings=1024, embedding_dim=8).to(device)
    criterion = MultiScaleMelSpectrogramLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        for batch_idx, (waveform, text) in enumerate(train_dataloader):
            # Get normalized mel spectogram
            mels = wav_to_mel_spectrogram(waveform.numpy(), sample_rate=sample_rate, n_mels = n_mels)
            mel_db = mel_normalize(mels)
            data = torch.from_numpy(mel_db).to(device)

            optimizer.zero_grad()
            output, _, _ = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}/{args.epochs}, Loss: {loss.item()}")

        # Log the loss value to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)

        # Save the model checkpoint
        if (epoch + 1) % 100 == 0:
            checkpoint_file = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Model checkpoint saved at {checkpoint_file}")

    # Save the final model
    model_file = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Final model saved at {model_file}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    args = parse_arguments()
    train(args)
