import os
import sys
import argparse
from datetime import datetime

# Add paths to import from external submodules and internal modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'submodule_project1'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'submodule_project2'))

# Import necessary libraries and modules
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import submodule_project1 as sp1
import submodule_project2 as sp2
from utils import some_util_function
from datasets import MyDataset
from models import MyModel
from losses import MyLoss

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment 1 Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./logs/experiment1", help="Output directory")

    return parser.parse_args()

def train(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    train_dataset = MyDataset(...)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(...).to(device)
    criterion = MyLoss(...).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch+1}/{args.epochs}, Loss: {loss.item()}")

        # Save the model checkpoint
        if (epoch + 1) % 10 == 0:
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
