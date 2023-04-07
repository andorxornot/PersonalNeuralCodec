import os
import sys
import argparse
from datetime import datetime

# Add paths to import from external submodules and internal modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'audiolm-pytorch'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'encodec'))

# Import necessary libraries and modules
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import audiolm_pytorch
import encodec
from utils.utils import *
from datasets.wav_folder_dataset import WavFolderDataset
from models.vqvae_model import VQVAE
from losses.mel_spectrogram_loss import MelSpectrogramLoss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment 1 Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./logs/experiment1", help="Output directory")

    return parser.parse_args()

def train(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    transform = RandomCrop(4096+512)
    train_dataset = WavFolderDataset("./data", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(in_channels=1, out_channels=1, hidden_channels=64, num_embeddings=10, embedding_dim=512).to(device)
    criterion = MelSpectrogramLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        for batch_idx, (data, text) in enumerate(train_dataloader):
            mels = wav_to_mel_spectrogram(data, sample_rate=22050, n_mels=80).to(device)
               
            optimizer.zero_grad()
            output, _, _ = model(mels)
            loss = criterion(output, mels)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}/{args.epochs}, Loss: {loss.item()}")

        # Save the model checkpoint
        if (epoch + 1) % 1000 == 0:
            checkpoint_file = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Model checkpoint saved at {checkpoint_file}")

    # Save the final model
    model_file = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Final model saved at {model_file}")

if __name__ == "__main__":
    args = parse_arguments()
    train(args)
