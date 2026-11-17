"""TRELLIS geometry-only extraction at256, same SDF bias/deformation/blend weights.

Color channels are omitted from extraction to reduce Mac memory; no geometry
channel changes. Sparse corner averages include each incident cube once.
"""
import torch
from .flexicubes import FlexiCubes
from .vendor.trellis.utils_cube import construct_dense_grid,sparse_cube2verts,get_defomed_verts,get_dense_attrs

def cube_fields(sparse,res=256):
    f=sparse.feats;coords=torch.as_tensor(sparse.coords,device=f.device,dtype=torch.long)
    attrs=torch.cat((f[:,:8].reshape(-1,8,1)-1/res,f[:,8:32].reshape(-1,8,3)),-1)
    return sparse_cube2verts(coords,attrs,training=torch.is_grad_enabled())

def extract_mesh(sparse,res=256,training=False):
    # Original dense FlexiCubes topology; CPU keeps large temporary grids off MPS.
    coords=torch.as_tensor(sparse.coords,dtype=torch.long);f=sparse.feats.cpu();sf=type(sparse)(f,sparse.coords)
    v_pos,attrs,consistency=cube_fields(sf,res);dense=get_dense_attrs(v_pos,attrs,res+1,True);weights=get_dense_attrs(coords,f[:,32:53],res,False)
    reg_v,reg_c=construct_dense_grid(res,'cpu');deformed=get_defomed_verts(reg_v,dense[:,1:4],res)
    v,faces,dev,_=FlexiCubes('cpu')(deformed,dense[:,0],reg_c,res,beta=weights[:,:12],alpha=weights[:,12:20],gamma_f=weights[:,20],training=training)
    return v,faces,dict(sparse_cubes=len(coords),sdf_negative=int((dense[:,0]<0).sum()),res=res,consistency=float(consistency) if not isinstance(consistency,float) else consistency)
