"""Eq.7 point-bone attention with two context sets; widths are pilot choices."""
import math
import torch
from torch import nn


class ContextBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.self_attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.surface_attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.skeleton_attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,x,surface,skeleton):
        for i,(attention,context) in enumerate([(self.self_attn,x),(self.surface_attn,surface),(self.skeleton_attn,skeleton)]):
            q=self.norms[i](x);kv=self.norms[i](context)
            x=x+attention(q,kv,kv,need_weights=False)[0]
        return x+self.ff(self.norms[3](x))


class SkinningDecoder(nn.Module):
    """Latent inputs here are explicit features, not silently claimed TRELLIS SLats.

    Sinusoidal positional encodings and memory masks are deferred; this is an
    initial equation/component implementation rather than author architecture.
    """
    def __init__(self,latent_dim=16,width=32,heads=4):
        super().__init__();self.points=nn.Linear(6,width);self.bones=nn.Linear(6,width)
        self.surface=nn.Linear(latent_dim,width);self.skeleton=nn.Linear(latent_dim,width)
        self.point_block=ContextBlock(width,heads);self.bone_block=ContextBlock(width,heads)
        self.point_out=nn.Linear(width,width);self.bone_out=nn.Linear(width,width);self.width=width
    def forward(self,points,normals,heads,tails,surface,skeleton):
        s=self.surface(surface);r=self.skeleton(skeleton)
        p=self.point_block(self.points(torch.cat((points,normals),-1)),s,r)
        b=self.bone_block(self.bones(torch.cat((heads,tails),-1)),s,r)
        return (self.point_out(p)@self.bone_out(b).transpose(-1,-2)/math.sqrt(self.width)).softmax(-1)


def soft_cross_entropy(prediction,target):
    if not torch.allclose(target.sum(-1),torch.ones_like(target.sum(-1)),atol=1e-5):raise ValueError('Target weights must sum to one')
    return -(target*prediction.clamp_min(1e-12).log()).sum(-1).mean()
