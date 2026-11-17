"""Mac-size coupled 3D occupancy-latent flows and sparse local-feature flows.

Original pretrained occupancy VAE supplies64^3 <->16^3x8. Two separate occupancy
networks observe both current noisy branches; local flows observe both generated
occupancy latents plus single-image spatial tokens. No GT coordinates at sample.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .decoders import position_encoding

def time_embed(t):
    f=2**torch.arange(8,device=t.device,dtype=t.dtype)*math.pi;p=t.reshape(-1,1)*f
    return torch.cat((p.sin(),p.cos()),-1)
def sample_occ_features(occ,coords):
    grid=coords[:,[2,1,0]][None,:,None,None]*2
    return F.grid_sample(occ,grid,align_corners=False)[:,:,:,0,0].transpose(1,2)[0]

class OccNet(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Conv3d(16,32,3,padding=1);self.down=nn.Conv3d(32,64,2,stride=2);self.norm=nn.GroupNorm(8,64);self.conv=nn.Sequential(nn.Conv3d(64,64,3,padding=1),nn.SiLU(),nn.Conv3d(64,64,3,padding=1));self.time=nn.Linear(16,64);self.image=nn.Linear(1024,64);self.attn=nn.MultiheadAttention(64,4,batch_first=True);self.up=nn.Conv3d(64,32,3,padding=1);self.out=nn.Conv3d(32,8,3,padding=1)
    def forward(self,x,t,image):
        skip=F.silu(self.input(x));h=self.down(skip);h=h+self.time(time_embed(t))[:,:,None,None,None];h=h+self.conv(F.silu(self.norm(h)));q=h.flatten(2).transpose(1,2);c=self.image(image);q=q+self.attn(q,c,c,need_weights=False)[0];h=q.transpose(1,2).reshape_as(h);h=self.up(F.interpolate(h,scale_factor=2,mode='nearest'));return self.out(F.silu(h+skip))
class LocalNet(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(8+24+16+16,64);self.image=nn.Linear(1024,64);self.attn=nn.MultiheadAttention(64,4,batch_first=True);self.ff=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,256),nn.SiLU(),nn.Linear(256,8))
    def forward(self,x,t,image,coords,occ):
        # F.grid_sample volume axes are z-y-x, while stored coordinates are x-y-z.
        local=sample_occ_features(occ,coords)
        q=self.input(torch.cat((x,position_encoding(coords),local,time_embed(t).expand(len(x),-1)),-1))[None];c=self.image(image);q=q+self.attn(q,c,c,need_weights=False)[0];return self.ff(q)[0]
class CoupledFlows(nn.Module):
    def __init__(self):
        super().__init__();self.occupancy=nn.ModuleDict({m:OccNet() for m in ['surface','rig']});self.local=nn.ModuleDict({m:LocalNet() for m in ['surface','rig']})
    def velocity(self,x,t,image):return torch.cat([self.occupancy[m](x,t,image) for m in ['surface','rig']],1)
    @torch.no_grad()
    def sample_occupancy(self,image,steps=25,cfg=5):
        x=torch.randn(1,16,16,16,16,device=image.device)
        for i in range(steps):
            t=x.new_tensor([i/steps]);v0=self.velocity(x,t,torch.zeros_like(image));v=self.velocity(x,t,image);x=x+(v0+cfg*(v-v0))/steps
        return x
    @torch.no_grad()
    def sample_local(self,image,occ,coords,mode,steps=25,cfg=5):
        z=torch.randn(len(coords),8,device=image.device)
        for i in range(steps):
            t=z.new_tensor([i/steps]);v0=self.local[mode](z,t,torch.zeros_like(image),coords,occ);v=self.local[mode](z,t,image,coords,occ);z=z+(v0+cfg*(v-v0))/steps
        return z
