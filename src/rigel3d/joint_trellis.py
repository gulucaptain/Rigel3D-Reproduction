"""Mac joint adaptation of pretrained SLat posterior heads, rig CA and rig decoders.

Frozen pretrained transformer/upsampling weights; all12 mesh blocks gain trainable
zero-initialized low-width rig cross-attention. Geometry uses small training
patches and oriented point-distance supervision, not original rendering losses.
"""
import torch
from torch import nn
from safetensors.torch import load_file
from .trellis_native import Sparse,load_pretrained
from .decoders import RigDecoders,position_encoding

class RigAdapters(nn.Module):
    def __init__(self,layers=12,width=64):
        super().__init__();self.query=nn.ModuleList([nn.Linear(768,width) for _ in range(layers)]);self.kv=nn.ModuleList([nn.Linear(8+24,2*width) for _ in range(layers)]);self.output=nn.ModuleList([nn.Linear(width,768) for _ in range(layers)])
        for m in self.output:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,i,x,context):
        z,c=context;kv=self.kv[i](torch.cat((z,position_encoding(c)),-1));k,v=kv.chunk(2,-1);q=self.query[i](torch.nn.functional.layer_norm(x.float(),(768,)))
        h=torch.nn.functional.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None])[0,0];return x+self.output[i](h).to(x.dtype)

class JointTrellis(nn.Module):
    def __init__(self,weights,device='mps'):
        super().__init__();self.surface_head=nn.Linear(768,16);self.rig_head=nn.Linear(768,16);state=load_file(str(weights/'slat_enc_swin8_B_64l8_fp16.safetensors'))
        for h in [self.surface_head,self.rig_head]:h.load_state_dict({k:state['out_layer.'+k] for k in ['weight','bias']})
        self.mesh=load_pretrained('mesh',weights,device).float();self.mesh.requires_grad_(False);self.mesh_adapter=RigAdapters();self.rig_decoders=RigDecoders(channels=8);self.to(device)
    def posterior(self,hidden,mode,sample=False):
        mu,lv=getattr(self,mode+'_head')(hidden).chunk(2,-1);z=mu+torch.randn_like(mu)*(.5*lv).exp() if sample else mu
        kl=.5*(mu.square()+lv.exp()-lv-1).mean();return z,mu,lv,kl
    def mesh_fields(self,z,coords,rz,rc):return self.mesh(Sparse(z,coords),self.mesh_adapter,(rz,rc))
    def trainable_state(self):return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('mesh.')}
    def load_trainable_state(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and all(k.startswith('mesh.') for k in missing)

class FullJointTrellis(JointTrellis):
    """Frozen surface encoder; actual full rig encoder and mesh weights trainable."""
    def __init__(self,weights,warm_state=None,device='mps'):
        super().__init__(weights,device)
        if warm_state is not None:self.load_trainable_state(warm_state)
        self.surface_head.requires_grad_(False);self.rig_head.requires_grad_(False)
        self.rig_encoder=load_pretrained('encoder',weights,device).float()
        self.rig_encoder.out_layer.load_state_dict(self.rig_head.state_dict())
        self.mesh.requires_grad_(True)
    def encode_rig(self,features,coords,sample=False):
        return self.rig_encoder(Sparse(features,coords),sample)
