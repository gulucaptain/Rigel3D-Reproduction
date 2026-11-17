"""Image-only generator. Ground-truth arrays are absent from this API."""
import numpy as np
import torch
from .data import fps_indices
from .flexicubes import regular_grid,FlexiCubes
from .representation import decode_tokens

@torch.no_grad()
def generate_asset(image,flows,field,decoders,latent_stats,steps=25,cfg=5):
    """image: frozen DINO feature of ONE view, shape [1,384]."""
    context,diag=flows.sample(image,steps=steps,cfg=cfg)
    if context is None:return None,diag
    tensors={}
    for mode in ['surface','rig']:
        d=context[mode];ix=fps_indices(d['coords'].cpu().numpy(),48)
        mu=torch.as_tensor(latent_stats[mode+'_mean'],device=image.device);sd=torch.as_tensor(latent_stats[mode+'_std'],device=image.device)
        tensors[mode]=(d['features'][ix][None]*sd+mu,d['coords'][ix][None])
    sf,sc=tensors['surface'];rf,rc=tensors['rig'];s,r=decoders.encoder(sf,sc,rf,rc)
    tokens,stop=decoders.skeleton.generate(s,r);joints,parents=decode_tokens(tokens.cpu().numpy())
    query,cubes=regular_grid(16,image.device);sdf=torch.cat([field(q[None],sf,sc,rf,rc)[0] for q in query.split(512)])
    # Upstream topology operations are exercised on CPU for portable Mac extraction;
    # learned field and all model inference above run on MPS.
    vertices,faces,_,_=FlexiCubes('cpu')(query.cpu(),sdf.cpu(),cubes.cpu(),16)
    v=vertices.numpy();f=faces.numpy();normal=np.zeros_like(v)
    if len(f):
        tri=v[f];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        for k in range(3):np.add.at(normal,f[:,k],fn)
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
    edges=np.flatnonzero(np.arange(len(parents))!=parents);bones=np.stack((joints[parents[edges]],joints[edges]),1) if len(edges) else np.zeros((0,2,3),np.float32)
    weights=[]
    if len(v) and len(bones):
        h=torch.as_tensor(bones[:,0],device=image.device,dtype=torch.float32)[None];t=torch.as_tensor(bones[:,1],device=image.device,dtype=torch.float32)[None]
        for start in range(0,len(v),512):
            p=torch.as_tensor(v[start:start+512],device=image.device)[None];n=torch.as_tensor(normal[start:start+512],device=image.device)[None]
            weights.append(decoders.skinning(p,n,h,t,s,r)[0].cpu().numpy())
    result=dict(vertices=v,faces=f,sdf=sdf.cpu().numpy(),tokens=tokens.cpu().numpy(),joints=joints,parents=parents,bones=bones,weights=np.concatenate(weights) if weights else np.zeros((len(v),0),np.float32),surface_coords=context['surface']['coords'].cpu().numpy(),rig_coords=context['rig']['coords'].cpu().numpy())
    diag.update(**stop,vertices=len(v),faces=len(f),generated_joints=len(joints),surface_cells=len(context['surface']['coords']),rig_cells=len(context['rig']['coords']),input_scope='one image DINO feature + noise; no target mesh, rig, skin, occupancy, count, prefix, or normalization at inference',failure='empty_mesh' if not len(f) else ('no_bones' if not len(bones) else None))
    return result,diag
