import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)

        return quantized, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, embedding_dim, 3, padding=1)
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, stride=2, padding=1)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, encoding_indices = self.vq(z_e)
        x_recon = self.decoder(z_q)

        return x_recon, z_e, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        _, encoding_indices = self.vq(z_e)
        return encoding_indices.view(x.size(0), -1)

    def decode(self, encoding_indices):
        z_q = self.vq.embeddings(encoding_indices).view(encoding_indices.size(0), self.vq.embedding_dim, -1, 1)
        x_recon = self.decoder(z_q)
        return x_recon
