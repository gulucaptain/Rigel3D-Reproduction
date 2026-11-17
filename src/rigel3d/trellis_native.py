"""Checkpoint-compatible TRELLIS SLat kernels for CPU/MPS.

Architecture/parameter names follow microsoft/TRELLIS @193e0502; native PyTorch
window SDPA and coordinate gather-convolution replace flash_attn/spconv only.
Batch-one sparse assets. No coordinate truncation or learned-weight adaptation.
"""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

class Sparse:
    def __init__(self,features,coords,cache=None):
        self.feats=features;self.coords=np.asarray(coords,dtype=np.int32);self.cache={} if cache is None else cache
        if self.coords.ndim!=2 or self.coords.shape[1]!=3 or len(self.coords)!=len(features):raise ValueError('Batch-one xyz/features required')
    def replace(self,x):return Sparse(x,self.coords,self.cache)
    def subdivide(self):
        c=(self.coords[:,None]*2+np.array(list(itertools.product(range(2),repeat=3)),np.int32)[None]).reshape(-1,3)
        return Sparse(self.feats[:,None].expand(-1,8,-1).reshape(-1,self.feats.shape[1]),c)

def absolute_pe(coords,channels,device):
    dim=channels//6;freq=1/(10000**(torch.arange(dim,device=device,dtype=torch.float32)/dim));x=torch.as_tensor(coords,device=device,dtype=torch.float32)
    phase=x[...,None]*freq;pe=torch.cat((phase.sin(),phase.cos()),-1).flatten(1)
    return F.pad(pe,(0,channels-pe.shape[1]))

def window_groups(coords,window=8,shift=0):
    _,inv=np.unique((coords+shift)//window,axis=0,return_inverse=True);order=np.argsort(inv,kind='stable');cuts=np.r_[0,np.cumsum(np.bincount(inv))]
    buckets={}
    for i in range(len(cuts)-1):
        ids=order[cuts[i]:cuts[i+1]];size=1<<(len(ids)-1).bit_length();buckets.setdefault(size,[]).append(ids)
    return buckets

def window_sdpa(qkv,coords,shift=0,cache=None):
    key=('window',shift);groups=(cache or {}).get(key)
    if groups is None:
        groups=window_groups(coords,8,shift)
        if cache is not None:cache[key]=groups
    n,_,heads,dim=qkv.shape;padded=torch.cat((qkv,qkv.new_zeros(1,3,heads,dim)));out=torch.zeros_like(qkv[:,0])
    for size,rows in groups.items():
        for start in range(0,len(rows),16):
            batch=rows[start:start+16];ix=np.full((len(batch),size),n,np.int64)
            for j,ids in enumerate(batch):ix[j,:len(ids)]=ids
            ix=torch.as_tensor(ix,device=qkv.device);packed=padded[ix].permute(0,2,3,1,4);q,k,v=packed.unbind(1)
            values=F.scaled_dot_product_attention(q,k,v,attn_mask=(ix<n)[:,None,None,:]).permute(0,2,1,3)
            valid=ix<n;out=out.index_copy(0,ix[valid],values[valid])
    return out.reshape(n,heads*dim)

class Attention(nn.Module):
    def __init__(self,width,heads,shift):
        super().__init__();self.to_qkv=nn.Linear(width,3*width);self.to_out=nn.Linear(width,width);self.heads=heads;self.shift=shift
    def forward(self,x):
        qkv=self.to_qkv(x.feats).reshape(len(x.coords),3,self.heads,-1)
        return x.replace(self.to_out(window_sdpa(qkv,x.coords,self.shift,x.cache)))
class FeedForward(nn.Module):
    def __init__(self,width):super().__init__();self.mlp=nn.Sequential(nn.Linear(width,width*4),nn.GELU(approximate='tanh'),nn.Linear(width*4,width))
    def forward(self,x):return self.mlp(x)
class Block(nn.Module):
    def __init__(self,width,heads,shift):super().__init__();self.attn=Attention(width,heads,shift);self.mlp=FeedForward(width)
    def forward(self,x):
        norm=lambda f:F.layer_norm(f.float(),(f.shape[-1],),eps=1e-6).to(f.dtype)
        h=x.feats+self.attn(x.replace(norm(x.feats))).feats
        return x.replace(h+self.mlp(norm(h)))

class SubMConv(nn.Module):
    def __init__(self,ci,co,k=3):
        super().__init__();self.k=k;self.conv=nn.Module();self.conv.weight=nn.Parameter(torch.empty(co,k,k,k,ci));self.conv.bias=nn.Parameter(torch.zeros(co));nn.init.normal_(self.conv.weight,std=(ci*k**3)**-.5)
    def forward(self,x):
        w=self.conv.weight;n=len(x.coords)
        if self.k==1:return x.replace(F.linear(x.feats,w[:,0,0,0],self.conv.bias))
        if ('conv',self.k) not in x.cache:
            # Collision-free integer-coordinate lookup; sentinels contribute zero.
            c=x.coords.astype(np.int64);stride=int(c.max()-c.min()+5);origin=c.min(0)-2
            code=lambda a:((a[:,0]-origin[0])*stride+(a[:,1]-origin[1]))*stride+a[:,2]-origin[2]
            keys=code(c);order=np.argsort(keys);sorted_keys=keys[order];indices=[]
            for off in itertools.product(range(-1,2),repeat=3):
                query=code(c+np.array(off));where=np.searchsorted(sorted_keys,query);valid=where<n;safe=np.minimum(where,n-1);valid&=sorted_keys[safe]==query;indices.append(np.where(valid,order[safe],n))
            x.cache['conv',self.k]=torch.as_tensor(np.stack(indices,1),device=x.feats.device)
        neighbors=x.cache['conv',self.k];padded=torch.cat((x.feats,x.feats.new_zeros(1,x.feats.shape[-1])));pieces=[]
        for start in range(0,n,4096):
            ix=neighbors[start:start+4096];h=self.conv.bias.expand(len(ix),-1)
            for j,off in enumerate(itertools.product(range(3),repeat=3)):h=h+F.linear(padded[ix[:,j]],w[(slice(None),)+off])
            pieces.append(h)
        return x.replace(torch.cat(pieces))
class SparseGN(nn.GroupNorm):
    def forward(self,x):
        h=F.group_norm(x.feats.float().T[None],self.num_groups,self.weight.float(),self.bias.float(),self.eps)[0].T
        return x.replace(h.to(x.feats.dtype))
class SparseSiLU(nn.Module):
    def forward(self,x):return x.replace(F.silu(x.feats))
class SubdivideBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__();self.act_layers=nn.Sequential(SparseGN(32,ci),SparseSiLU());self.out_layers=nn.Sequential(SubMConv(ci,co),SparseGN(32,co),SparseSiLU(),SubMConv(co,co));self.skip_connection=SubMConv(ci,co,1)
    def forward(self,x):
        h=self.act_layers(x).subdivide();skip=x.subdivide();skip.cache=h.cache;h=self.out_layers(h)
        return h.replace(h.feats+self.skip_connection(skip).feats)

class SLatBase(nn.Module):
    def __init__(self,ci,width=768,blocks=12,heads=12):
        super().__init__();self.input_layer=nn.Linear(ci,width);self.blocks=nn.ModuleList([Block(width,heads,4*(i%2)) for i in range(blocks)]);self.width=width
    def trunk(self,x,rig_adapter=None,rig_context=None):
        h=self.input_layer(x.feats)+absolute_pe(x.coords,self.width,x.feats.device)
        dtype=self.blocks[0].attn.to_qkv.weight.dtype;h=x.replace(h.to(dtype))
        for i,b in enumerate(self.blocks):
            h=b(h)
            if rig_adapter is not None:h=h.replace(rig_adapter(i,h.feats,rig_context))
        return h
class SLatEncoder(SLatBase):
    def __init__(self):super().__init__(1024);self.out_layer=nn.Linear(768,16)
    def forward(self,x,sample=False):
        h=self.trunk(x);f=F.layer_norm(h.feats.float(),(self.width,));mu,lv=self.out_layer(f).chunk(2,-1)
        z=mu+torch.randn_like(mu)*(.5*lv).exp() if sample else mu
        return x.replace(z),mu,lv
class SLatMeshDecoder(SLatBase):
    def __init__(self):
        super().__init__(8);self.upsample=nn.ModuleList([SubdivideBlock(768,192),SubdivideBlock(192,96)]);self.out_layer=nn.Linear(96,101)
    def forward(self,x,rig_adapter=None,rig_context=None):
        h=self.trunk(x,rig_adapter,rig_context)
        for block in self.upsample:h=block(h)
        return h.replace(self.out_layer(h.feats.float()))

def load_pretrained(kind,folder,device='cpu'):
    names={'encoder':'slat_enc_swin8_B_64l8_fp16','mesh':'slat_dec_mesh_swin8_B_64l8m256c_fp16'}
    cls=SLatEncoder if kind=='encoder' else SLatMeshDecoder;model=cls();weights=load_file(str(Path(folder)/(names[kind]+'.safetensors')))
    # assign=True retains checkpoint's mixed fp16 torso/fp32 IO and norm dtypes.
    model.load_state_dict(weights,strict=True,assign=True);return model.to(device)
