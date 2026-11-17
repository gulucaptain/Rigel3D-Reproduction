"""Native coordinate-aware sparse VAE pilot, not a port of pretrained TRELLIS.

Submanifold seven-tap axial convolution preserves occupied 64-grid coordinates.
Missing neighbors contribute zero. KL uses N(mu,diag(exp(logvar))) versus N(0,I).
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OFFSETS=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])

def neighbor_indices(coords):
    c=np.asarray(coords,dtype=int)
    if len(np.unique(c,axis=0))!=len(c):raise ValueError('Duplicate sparse coordinates')
    lookup={tuple(v):i for i,v in enumerate(c)}
    return np.asarray([[lookup.get(tuple(v+o),len(c)) for o in OFFSETS] for v in c],dtype=np.int64)

class SparseConv(nn.Module):
    def __init__(self,width):
        super().__init__();self.weight=nn.Parameter(torch.randn(7,width,width)/(7*width)**.5);self.bias=nn.Parameter(torch.zeros(width))
    def forward(self,x,neighbors):
        # Seven independent matrices, not a coordinate-blind feature MLP.
        padded=torch.cat((x,x.new_zeros(1,x.shape[-1])))
        return sum(padded[neighbors[:,i]]@self.weight[i] for i in range(7))+self.bias

class SparseBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width);self.conv=SparseConv(width)
    def forward(self,x,neighbors):return x+F.gelu(self.conv(self.norm(x),neighbors))

class SparseVAE(nn.Module):
    def __init__(self,channels=384,width=32,latent=8):
        super().__init__();self.input=nn.Linear(channels,width);self.enc=nn.ModuleList([SparseBlock(width) for _ in range(2)]);self.stats=nn.Linear(width,2*latent)
        self.latent=nn.Linear(latent,width);self.dec=nn.ModuleList([SparseBlock(width) for _ in range(2)]);self.output=nn.Linear(width,channels)
    def encode(self,x,neighbors):
        h=self.input(x)
        for b in self.enc:h=b(h,neighbors)
        mu,logvar=self.stats(h).chunk(2,-1);return mu,logvar.clamp(-12,8)
    def decode(self,z,neighbors):
        h=self.latent(z)
        for b in self.dec:h=b(h,neighbors)
        return self.output(h)
    def forward(self,x,neighbors,sample=True):
        mu,lv=self.encode(x,neighbors);z=mu+torch.randn_like(mu)*(.5*lv).exp() if sample else mu
        pred=self.decode(z,neighbors);kl=.5*(mu.square()+lv.exp()-lv-1).mean()
        return pred,kl,mu
