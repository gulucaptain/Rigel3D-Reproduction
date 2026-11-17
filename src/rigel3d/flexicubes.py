"""Regular grid adapter for pinned upstream FlexiCubes, CPU and MPS."""
import torch
from .vendor.flexicubes.flexicubes import FlexiCubes

def regular_grid(resolution,device='cpu'):
    r=resolution;axis=torch.linspace(-.5,.5,r+1,device=device)
    vertices=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    base=torch.stack(torch.meshgrid(*(torch.arange(r,device=device),)*3,indexing='ij'),-1).reshape(-1,3)
    corners=torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]],device=device)
    c=base[:,None]+corners;idx=c[...,0]*(r+1)**2+c[...,1]*(r+1)+c[...,2]
    return vertices,idx.long()
