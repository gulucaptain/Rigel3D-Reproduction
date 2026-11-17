"""Strict image-only pretrained-TRELLIS-based rig generator.

No target data loader or GT arguments. Runtime spatial grids come exclusively
from decoded predicted occupancy. Budget violations are explicit failures,
never hidden top-k truncation or GT-count/voxel substitutions.
"""
import json
import numpy as np
import torch
from safetensors.torch import load_file
from .vendor.trellis.occupancy import SparseStructureDecoder
from .data import fps_indices
from .trellis_mesh import extract_mesh
from .representation import decode_tokens,rig_voxels

def load_occupancy_decoder(weights,device='mps'):
    name='ss_dec_conv3d_16l8_fp16';config=json.load(open(weights/(name+'.json')));m=SparseStructureDecoder(**config['args']);m.load_state_dict(load_file(str(weights/(name+'.safetensors'))),strict=True,assign=True);m.convert_to_fp32();return m.to(device).eval()

@torch.no_grad()
def generate_skeleton(decoder,s,r,threshold=.5):
    previous=torch.empty(1,0,4,device=s.device,dtype=torch.long);stopped=False;probabilities=[]
    for k in range(decoder.max_joints):
        out=decoder(previous,s,r);prob=float(out['stop'][0,-1].softmax(-1)[1].cpu());probabilities.append(prob)
        if k>0 and prob>=threshold:stopped=True;break
        token=torch.cat((out['coords'][0,-1].argmax(-1),out['parent'][0,-1].argmax()[None]))[None,None];previous=torch.cat((previous,token),1)
    return previous[0],dict(eos=stopped,capped=not stopped,stop_threshold=threshold,eos_probabilities=probabilities)

@torch.no_grad()
def decode_rigged_asset(joint,surface_z,surface_coords,rig_z,rig_coords,skin=True,stop_threshold=.5):
    device=surface_z.device;si=fps_indices(surface_coords,48);ri=fps_indices(rig_coords,48);sc=torch.tensor(((surface_coords+.5)/64-.5).astype('float32'),device=device);rc=torch.tensor(((rig_coords+.5)/64-.5).astype('float32'),device=device)
    s,r=joint.rig_decoders.encoder(surface_z[si][None],sc[si][None],rig_z[ri][None],rc[ri][None]);tokens,diag=generate_skeleton(joint.rig_decoders.skeleton,s,r,stop_threshold);joints,parents=decode_tokens(tokens.cpu().numpy())
    fields=joint.mesh_fields(surface_z,surface_coords,rig_z[ri],rc[ri]);v,f,extract=extract_mesh(fields);v=v.numpy();f=f.numpy();normal=np.zeros_like(v)
    if len(f):
        tri=v[f];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);
        for k in range(3):np.add.at(normal,f[:,k],fn)
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
    edges=np.flatnonzero(parents!=np.arange(len(parents)));bones=np.stack((joints[parents[edges]],joints[edges]),1) if len(edges) else np.zeros((0,2,3),np.float32);weights=[]
    if skin and len(v) and len(bones):
        h=torch.tensor(bones[:,0],device=device,dtype=torch.float32)[None];t=torch.tensor(bones[:,1],device=device,dtype=torch.float32)[None]
        # Batch independent512-query chunks; same approximation as previous pilot.
        full=(len(v)//512)*512
        for start in range(0,full,4096):
            end=min(start+4096,full);batch=(end-start)//512;p=torch.tensor(v[start:end],device=device).reshape(batch,512,3);n=torch.tensor(normal[start:end],device=device).reshape(batch,512,3)
            w=joint.rig_decoders.skinning(p,n,h.expand(batch,-1,-1),t.expand(batch,-1,-1),s.expand(batch,-1,-1),r.expand(batch,-1,-1));weights.append(w.reshape(-1,len(bones)).cpu().numpy())
        if full<len(v):weights.append(joint.rig_decoders.skinning(torch.tensor(v[full:],device=device)[None],torch.tensor(normal[full:],device=device)[None],h,t,s,r)[0].cpu().numpy())
    weight=np.concatenate(weights) if weights else np.zeros((len(v),0),np.float32)
    if len(bones):
        ar_coords=rig_voxels(bones[:,0],bones[:,1],np.ones((len(bones),1)),64)[0];ar_set={tuple(c) for c in ar_coords};latent_set={tuple(c) for c in rig_coords};diag['ar_vs_latent_rig_iou']=len(ar_set&latent_set)/max(len(ar_set|latent_set),1);diag['ar_rig_voxels']=len(ar_set);diag['zero_length_bones']=int((np.linalg.norm(bones[:,1]-bones[:,0],axis=-1)<1e-6).sum())
    result=dict(vertices=v,faces=f,joints=joints,parents=parents,tokens=tokens.cpu().numpy(),bones=bones,weights=weight,surface_coords=surface_coords,rig_coords=rig_coords,surface_latents=surface_z.cpu().numpy(),rig_latents=rig_z.cpu().numpy())
    diag.update(vertices=len(v),faces=len(f),generated_joints=len(joints),extraction=extract,failure='empty_mesh' if not len(f) else ('no_generated_bones' if not len(bones) else None),skin_rows_sum_max_error=float(np.abs(weight.sum(-1)-1).max()) if weight.shape[1] else None)
    return result,diag

@torch.no_grad()
def generate_image_only(image_tokens,flows,joint,occupancy_decoder,stats,steps=25,cfg=5,max_surface=16000,max_rig=12000):
    """Accept ONE image token tensor, trained models and train-only statistics."""
    if image_tokens.shape!=(1,64,1024):raise ValueError('One pooled8x8 DINO-L image condition required')
    state=flows.sample_occupancy(image_tokens,steps,cfg);coords={};occupancy={};diag=dict(steps=steps,cfg=cfg,mesh_attempted=False,input_scope='One image tokens + noise + learned weights/train statistics only. No target occupancy/geometry/rig/count/prefix or normalization read at runtime.',budget_policy='Fail explicitly above fixed16000 surface/12000 rig cells; no truncation')
    for i,mode in enumerate(['surface','rig']):
        mu=torch.as_tensor(stats[mode+'_occ_mean'],device=state.device)[None,:,None,None,None];std=torch.as_tensor(stats[mode+'_occ_std'],device=state.device)[None,:,None,None,None];z=state[:,i*8:(i+1)*8]*std+mu;logits=occupancy_decoder(z);mask=logits[0,0]>0;occupancy[mode]=mask.cpu().numpy();coords[mode]=torch.nonzero(mask).cpu().numpy();diag[mode+'_voxels']=len(coords[mode])
    partial=dict(surface_occupancy=occupancy['surface'],rig_occupancy=occupancy['rig'],occupancy_latents=state.cpu().numpy())
    for mode,budget in [('surface',max_surface),('rig',max_rig)]:
        if not len(coords[mode]):diag['failure']='empty_predicted_'+mode+'_occupancy';return partial,diag
        if len(coords[mode])>budget:diag['failure']=mode+'_occupancy_budget_exceeded';return partial,diag
    latents={}
    for mode in ['surface','rig']:
        c=torch.tensor(((coords[mode]+.5)/64-.5).astype('float32'),device=state.device);z=flows.sample_local(image_tokens,state,c,mode,steps,cfg);mu=torch.as_tensor(stats[mode+'_local_mean'],device=state.device);std=torch.as_tensor(stats[mode+'_local_std'],device=state.device);latents[mode]=z*std+mu
    diag['mesh_attempted']=True
    result,decoded=decode_rigged_asset(joint,latents['surface'],coords['surface'],latents['rig'],coords['rig']);result.update(partial);diag.update(decoded);return result,diag
