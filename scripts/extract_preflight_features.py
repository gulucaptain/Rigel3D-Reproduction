"""Official frozen DINOv2 patch features on one original training mesh.

Feature-preparation preflight only; not used to relabel the geometry-only runs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
root=Path(__file__).resolve().parents[1];base=root/'data/external/generation_preflight'
torch.set_num_threads(4);torch.hub.set_dir(str(root/'data/external/torchhub'))
revision=json.loads((root/'sources/dinov2-revision.json').read_text())['commit']
model=torch.hub.load('facebookresearch/dinov2:'+revision,'dinov2_vits14',pretrained=True,trust_repo=True).eval().to('mps')
images=np.stack([np.asarray(Image.open(base/f'view_{i}.png').convert('RGB')) for i in range(4)])
x=torch.tensor(images.transpose(0,3,1,2),device='mps',dtype=torch.float32)/255
x=(x-torch.tensor([.485,.456,.406],device='mps')[None,:,None,None])/torch.tensor([.229,.224,.225],device='mps')[None,:,None,None]
with torch.inference_mode():features=model.forward_features(x)['x_norm_patchtokens'].cpu().numpy()
if not np.isfinite(features).all():raise ValueError('Nonfinite DINO features')
np.savez_compressed(base/'dinov2_features.npz',patch_features=features)
if (base/'vertex_visibility.npz').exists():
    v=dict(np.load(base/'vertex_visibility.npz'));n=v['pixels'].shape[1]
    grid=torch.tensor(v['pixels']/224*2-1).reshape(4,n,1,2)
    maps=torch.tensor(features).reshape(4,16,16,384).permute(0,3,1,2)
    projected=torch.nn.functional.grid_sample(maps,grid,align_corners=False)[...,0].permute(0,2,1).numpy()
    visible=v['visible']&(np.abs(v['pixels']/224*2-1)<1).all(-1)
    count=visible.sum(0);observed=count>0
    if not observed.any():raise ValueError('No visible mesh vertices; projection/pose alignment failed')
    vf=(projected*visible[...,None]).sum(0)/np.maximum(count[:,None],1)
    points=np.load(base/'original_mesh_vertices.npz')['points']
    coords=np.floor((points[observed]+.5)*64).astype(int)
    unique,idx=np.unique(coords,axis=0,return_inverse=True);agg=np.zeros((len(unique),384));vc=np.zeros(len(unique))
    np.add.at(agg,idx,vf[observed]);np.add.at(vc,idx,1);agg/=vc[:,None]
    np.savez_compressed(base/'surface_voxel_features.npz',coords=unique,features=agg.astype(np.float32))
    projection_report=dict(vertices=n,observed_vertices=int(observed.sum()),observed_fraction=float(observed.mean()),
        active_surface_voxels=len(unique),feature_channels=384,visibility='Blender ray cast per original mesh vertex, tolerance 2e-4; four-view patch bilinear average',
        scope='Actual sparse surface feature input only; no learned TRELLIS encoder/SLats')
else:projection_report=None
report=json.loads((root/'results/round1/generation_preflight.json').read_text())
checkpoints=list((root/'data/external/torchhub/checkpoints').glob('*.pth'))
report.update(dinov2=dict(repository_commit=revision,model='dinov2_vits14',device='mps',shape=list(features.shape),
    feature_mean=float(features.mean()),feature_std=float(features.std()),
    checkpoints=[dict(filename=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in checkpoints],
    scope='Real original-train-asset views and pretrained features only; no Sparse VAE, FlexiCubes or generator training; not fed into completed geometry-feature runs'))
report['surface_feature_projection']=projection_report
(root/'results/round1/generation_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['dinov2']))
