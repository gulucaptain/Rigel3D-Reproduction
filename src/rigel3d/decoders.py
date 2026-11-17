"""Trainable local decoders: causal joint tuples, constrained parents, masked skinning.

Widths/depths/PE choice are explicit pilot assumptions, not author checkpoint shapes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def position_encoding(x,bands=4):
    frequencies=2**torch.arange(bands,device=x.device,dtype=x.dtype)*math.pi
    phase=x[...,None]*frequencies
    return torch.cat((phase.sin(),phase.cos()),-1).flatten(-2)


class DualBlock(nn.Module):
    def __init__(self,width=64,heads=4,causal=False):
        super().__init__();self.causal=causal
        self.attention=nn.ModuleList([nn.MultiheadAttention(width,heads,batch_first=True) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,x,surface,rig,x_pad=None,surface_pad=None,rig_pad=None):
        causal=torch.ones((x.shape[1],x.shape[1]),device=x.device,dtype=torch.bool).triu(1) if self.causal else None
        for i,(context,pad) in enumerate([(x,x_pad),(surface,surface_pad),(rig,rig_pad)]):
            q=self.norms[i](x);kv=self.norms[i](context)
            x=x+self.attention[i](q,kv,kv,key_padding_mask=pad,attn_mask=causal if i==0 else None,need_weights=False)[0]
        return x+self.ff(self.norms[3](x))


class ContextEncoder(nn.Module):
    """Small learned projection of explicit 64-grid context, not TRELLIS Sparse VAE."""
    def __init__(self,width=64,channels=16):
        super().__init__();self.surface=nn.Sequential(nn.Linear(channels+24,width),nn.GELU(),nn.Linear(width,width))
        self.rig=nn.Sequential(nn.Linear(channels+24,width),nn.GELU(),nn.Linear(width,width))
    def forward(self,surface_features,surface_coords,rig_features,rig_coords):
        return (self.surface(torch.cat((surface_features,position_encoding(surface_coords)),-1)),
            self.rig(torch.cat((rig_features,position_encoding(rig_coords)),-1)))


class SkeletonDecoder(nn.Module):
    def __init__(self,width=64,heads=4,layers=2,resolution=128,max_joints=96):
        super().__init__();self.resolution=resolution;self.max_joints=max_joints
        self.coord_embed=nn.ModuleList([nn.Embedding(resolution,width) for _ in range(3)])
        self.parent_embed=nn.Embedding(max_joints,width);self.sequence_embed=nn.Embedding(max_joints+1,width)
        self.bos=nn.Parameter(torch.randn(width)*.02)
        self.blocks=nn.ModuleList([DualBlock(width,heads,True) for _ in range(layers)])
        self.norm=nn.LayerNorm(width);self.coordinate=nn.Linear(width,3*resolution)
        self.parent=nn.Linear(width,max_joints);self.stop=nn.Linear(width,2)
    def forward(self,previous,surface,rig,surface_pad=None,rig_pad=None,sequence_pad=None):
        # previous contains only past joint tuples. Output k predicts tuple k.
        batch,n,_=previous.shape
        embedded=sum(e(previous[...,i]) for i,e in enumerate(self.coord_embed))+self.parent_embed(previous[...,3])
        x=torch.cat((self.bos[None,None].expand(batch,1,-1),embedded),1)
        indices=torch.arange(n+1,device=x.device)
        x=x+self.sequence_embed(indices)[None]
        for block in self.blocks:x=block(x,surface,rig,sequence_pad,surface_pad,rig_pad)
        x=self.norm(x);parent=self.parent(x)
        invalid=torch.arange(self.max_joints,device=x.device)[None]>=indices[:,None].clamp_min(1)
        parent=parent.masked_fill(invalid[None],-1e4)
        return dict(coords=self.coordinate(x).reshape(batch,n+1,3,self.resolution),parent=parent,stop=self.stop(x))
    @torch.no_grad()
    def generate(self,surface,rig,surface_pad=None,rig_pad=None):
        if surface.shape[0]!=1:raise ValueError('Explicit single-asset greedy generation')
        previous=torch.empty((1,0,4),device=surface.device,dtype=torch.long);stopped=False
        for k in range(self.max_joints):
            out=self(previous,surface,rig,surface_pad,rig_pad)
            # At least one root joint is required.
            if k>0 and int(out['stop'][0,-1].argmax())==1:stopped=True;break
            coord=out['coords'][0,-1].argmax(-1);parent=out['parent'][0,-1].argmax()
            token=torch.cat((coord,parent[None]))[None,None]
            previous=torch.cat((previous,token),1)
        return previous[0],dict(eos=stopped,capped=not stopped)


def skeleton_loss(out,target,lengths,balanced_stop=False):
    b,l,_=target.shape;k=torch.arange(l,device=target.device)[None];valid=k<lengths[:,None]
    coords=F.cross_entropy(out['coords'][:,:l][valid].reshape(-1,out['coords'].shape[-1]),target[valid][:,:3].reshape(-1))
    parent=F.cross_entropy(out['parent'][:,:l][valid],target[valid][:,3])
    stop_valid=torch.arange(out['stop'].shape[1],device=target.device)[None]<=lengths[:,None]
    stop_target=(torch.arange(out['stop'].shape[1],device=target.device)[None]==lengths[:,None]).long()
    logits=out['stop'][stop_valid];labels=stop_target.expand(b,-1)[stop_valid]
    if balanced_stop:
        # Explicit diagnostic alternative; original paper does not disclose
        # EOS weighting. Balance per-class means, never tune using test metrics.
        ce=F.cross_entropy(logits,labels,reduction='none')
        stop=.5*(ce[labels==0].mean()+ce[labels==1].mean())
    else:stop=F.cross_entropy(logits,labels)
    return coords+parent+stop,dict(coords=coords,parent=parent,stop=stop)


class SkinningDecoder(nn.Module):
    def __init__(self,width=64,heads=4,layers=2):
        super().__init__();self.width=width
        self.point_in=nn.Linear(6+24,width);self.bone_in=nn.Linear(6+48,width)
        self.point_blocks=nn.ModuleList([DualBlock(width,heads) for _ in range(layers)])
        self.bone_blocks=nn.ModuleList([DualBlock(width,heads) for _ in range(layers)])
        self.point_out=nn.Linear(width,width);self.bone_out=nn.Linear(width,width)
    def forward(self,points,normals,heads,tails,surface,rig,bone_pad=None,surface_pad=None,rig_pad=None):
        p=self.point_in(torch.cat((points,normals,position_encoding(points)),-1))
        b=self.bone_in(torch.cat((heads,tails,position_encoding(heads),position_encoding(tails)),-1))
        for pb,bb in zip(self.point_blocks,self.bone_blocks):
            p=pb(p,surface,rig,None,surface_pad,rig_pad);b=bb(b,surface,rig,bone_pad,surface_pad,rig_pad)
        logits=self.point_out(p)@self.bone_out(b).transpose(-1,-2)/math.sqrt(self.width)
        if bone_pad is not None:logits=logits.masked_fill(bone_pad[:,None],-1e4)
        return logits.softmax(-1)


class RigDecoders(nn.Module):
    def __init__(self,width=64,layers=2,max_joints=96,channels=16):
        super().__init__();self.encoder=ContextEncoder(width,channels)
        self.skeleton=SkeletonDecoder(width=width,layers=layers,max_joints=max_joints)
        self.skinning=SkinningDecoder(width=width,layers=layers)
