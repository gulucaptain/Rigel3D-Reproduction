"""Small image-conditioned four-flow and field-decoder pilot.

Deliberate scale changes: direct 16^3 occupancy flows (no occupancy compression VAE),
MLPs instead of original sparse flow transformers; no pretrained TRELLIS mesh decoder.
Inference methods accept image features/noise, never target geometry or rig.
"""
import math
import torch
from torch import nn
from .decoders import position_encoding

class VectorFlow(nn.Module):
    def __init__(self,dim,width=128):
        super().__init__();self.x=nn.Linear(dim,width);self.condition=nn.Linear(384,width);self.time=nn.Linear(16,width)
        self.net=nn.Sequential(nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,dim))
    def forward(self,x,t,image):
        freq=2**torch.arange(8,device=x.device)*math.pi;p=t[:,None]*freq
        return self.net(self.x(x)+self.condition(image)+self.time(torch.cat((p.sin(),p.cos()),-1)))

class LocalFlow(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(8+24+16+384,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,8))
    def forward(self,x,t,image,coords):
        freq=2**torch.arange(8,device=x.device)*math.pi;p=t[:,None]*freq;pe=torch.cat((p.sin(),p.cos()),-1)
        return self.net(torch.cat((x,position_encoding(coords),pe.expand(len(x),-1),image.expand(len(x),-1)),-1))

class FourFlows(nn.Module):
    def __init__(self):
        super().__init__();self.occupancy=nn.ModuleDict({m:VectorFlow(16**3) for m in ['surface','rig']});self.local=nn.ModuleDict({m:LocalFlow() for m in ['surface','rig']})
    @torch.no_grad()
    def sample(self,image,steps=25,cfg=5,max_cells=256):
        # No GT position/count/prefix argument. All lattice positions come from predicted occupancy.
        result={};counts={}
        for mode in ['surface','rig']:
            x=torch.randn(1,16**3,device=image.device)
            for i in range(steps):
                t=x.new_tensor([i/steps]);v0=self.occupancy[mode](x,t,torch.zeros_like(image));v=self.occupancy[mode](x,t,image);x=x+(v0+cfg*(v-v0))/steps
            ids=torch.nonzero(x[0]>0).flatten();counts[mode+'_positive_cells_before_cap']=len(ids);counts[mode+'_cell_cap_hit']=len(ids)>max_cells
            if len(ids)>max_cells:ids=ids[x[0,ids].topk(max_cells).indices]
            if len(ids)==0:return None,dict(failure='empty_predicted_'+mode+'_occupancy',**counts)
            coords=torch.stack((ids//256,(ids//16)%16,ids%16),-1).float();coords=(coords+.5)/16-.5
            z=torch.randn(len(ids),8,device=image.device)
            for i in range(steps):
                t=z.new_tensor([i/steps]);v0=self.local[mode](z,t,torch.zeros_like(image),coords);v=self.local[mode](z,t,image,coords);z=z+(v0+cfg*(v-v0))/steps
            result[mode]=dict(coords=coords,features=z)
        return result,dict(failure=None,steps=steps,cfg=cfg,max_cells=max_cells,**counts)

class FieldDecoder(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.query=nn.Linear(24,width);self.surface=nn.Linear(8+24,width);self.rig=nn.Linear(8+24,width)
        self.attn=nn.ModuleList([nn.MultiheadAttention(width,4,batch_first=True) for _ in range(2)]);self.out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self,points,sf,sc,rf,rc):
        x=self.query(position_encoding(points));contexts=[self.surface(torch.cat((sf,position_encoding(sc)),-1)),self.rig(torch.cat((rf,position_encoding(rc)),-1))]
        for a,c in zip(self.attn,contexts):x=x+a(x,c,c,need_weights=False)[0]
        return .15*self.out(x).squeeze(-1).tanh()
