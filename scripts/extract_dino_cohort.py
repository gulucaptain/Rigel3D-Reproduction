"""Frozen real DINOv2 features, visibility-aware multiview projection and Eq.2 rig pooling."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scipy.spatial import cKDTree
from rigel3d.data import pool_cells
from rigel3d.representation import rig_voxels
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.hub.set_dir(str(root/'data/external/torchhub'))
rev=json.loads((root/'sources/dinov2-revision.json').read_text())['commit'];model=torch.hub.load('facebookresearch/dinov2:'+rev,'dinov2_vits14',pretrained=True,trust_repo=True).eval().to('mps')
records=json.loads((root/'results/round2/mesh_alignment.json').read_text())['assets'];out=[]
for row in records:
    if row.get('alignment_status')!='accepted':continue
    folder=root/'data/external/dino_cohort'/Path(row['filename']).stem;target=folder/'features.npz'
    if target.exists():out.append(json.loads((folder/'features.json').read_text()));continue
    images=np.stack([np.asarray(Image.open(folder/f'view_{i}.png').convert('RGB')) for i in range(4)])
    x=torch.tensor(images.transpose(0,3,1,2).copy(),device='mps',dtype=torch.float32)/255
    x=(x-torch.tensor([.485,.456,.406],device='mps')[None,:,None,None])/torch.tensor([.229,.224,.225],device='mps')[None,:,None,None]
    with torch.inference_mode():patch=model.forward_features(x)['x_norm_patchtokens'].cpu()
    pr=np.load(folder/'projection.npz');grid=torch.tensor(pr['pixels']/224*2-1).reshape(4,-1,1,2)
    projected=torch.nn.functional.grid_sample(patch.reshape(4,16,16,384).permute(0,3,1,2),grid,align_corners=False)[...,0].permute(0,2,1).numpy()
    visible=pr['visible'];count=visible.sum(0);observed=count>0
    if observed.sum()<100:raise ValueError('Insufficient visible points '+row['filename'])
    d=np.load(root/'data/external/prepared'/row['filename']);point_features=(projected*visible[...,None]).sum(0)/np.maximum(count[:,None],1)
    # Explicitly fill occluded queries from nearest visible SURFACE point, no GT labels.
    missing=~observed;nearest=cKDTree(d['points'][observed]).query(d['points'][missing])[1];point_features[missing]=point_features[observed][nearest]
    sc,sf=pool_cells(np.floor((d['points']+.5)*64).astype(int).clip(0,63),point_features)
    bf=d['weights'].T@point_features/(d['weights'].sum(0)[:,None]+1e-8)
    rc,rf,_=rig_voxels(d['bones'][:,0],d['bones'][:,1],bf,64)
    np.savez_compressed(target,surface_coords=sc,surface_features=sf.astype('float32'),rig_coords=rc,rig_features=rf.astype('float32'),image_features=patch.numpy(),point_features=point_features.astype('float32'))
    rec=dict(filename=row['filename'],split=row['split'],observed_fraction=float(observed.mean()),surface_voxels=len(sc),rig_voxels=len(rc),sha256=hashlib.sha256(target.read_bytes()).hexdigest());out.append(rec);(folder/'features.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
weights=root/'data/external/torchhub/checkpoints/dinov2_vits14_pretrain.pth'
report=dict(model='dinov2_vits14',repository_commit=rev,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),channels=384,views=4,image_size=224,renderer='Blender Workbench neutral material; no original textures',occlusion='orthographic first-hit distance < .02; unseen points nearest visible surface feature',scope='Surface uses real images; rig feature/voxel coordinates use GT bones and skin weights: reconstruction only',assets=out)
(root/'results/round2/dino_features.json').write_text(json.dumps(report,indent=2)+'\n')
