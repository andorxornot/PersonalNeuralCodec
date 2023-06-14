import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from quantization  import ResidualVectorQuantizer
from msstftd import MultiScaleSTFTDiscriminator
# Generator
import math
import numpy as np
import random
from modules.seanet import SEANetEncoder, SEANetDecoder
class SoundStream(nn.Module):
    def __init__(self, n_filters, D, 
                 target_bandwidths=[1, 1.5, 2, 4, 6, 12],
                 ratios=[8, 5, 4, 2],
                 sample_rate=16000,
                 bins=1024,
                 normalize=False):
        super().__init__()
        self.hop_length = np.prod(ratios) # 计算乘积
        # print('self.hop_length ', self.hop_length)
        self.encoder = SEANetEncoder(n_filters= n_filters, dimension=D, ratios=ratios)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios)) # 75
        self.bits_per_codebook = int(math.log2(bins))
        self.target_bandwidths = target_bandwidths
        self.quantizer = ResidualVectorQuantizer(dimension=D, n_q=n_q, bins=bins)
        self.decoder = SEANetDecoder(n_filters= n_filters, dimension=D, ratios=ratios)
    
    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
    def forward(self, x):
        e = self.encoder(x)
        bw = self.target_bandwidths[random.randint(0, 5)] # [0, 4]
        #bw = self.target_bandwidths[-1]
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        o = self.decoder(quantized)
        # print('o ', o.shape)
        # assert 1==2
        return o, commit_loss, None
    def encode(self,x, target_bw=None, st=None):
        e = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        if st is None:
            st = 0
        codes = self.quantizer.encode(e, self.frame_rate, bw, st)
        return codes
    def decode(self, codes):
        quantized = self.quantizer.decode(codes)
        o = self.decoder(quantized)
        return o
