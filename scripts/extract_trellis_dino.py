"""Real 1024-channel DINO-L/reg, official TRELLIS x_prenorm preprocessing.

Uses existing four calibrated original-asset views resized to518; their native
render resolution224 and neutral material remain explicit input limitations.
Sampled points projected in all views, then voxel-pooled; differs from direct voxel-center projection.
"""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from rigel3d.data import pool_cells
from rigel3d.representation import rig_voxels
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.hub.set_dir(str(root/'data/external/torchhub'));rev=json.load(open(root/'sources/dinov2-revision.json'))['commit'];model=torch.hub.load('facebookresearch/dinov2:'+rev,'dinov2_vitl14_reg',pretrained=True,trust_repo=True).to('mps').eval()
records=json.load(open(root/'results/round2/dino_features.json'))['assets'];rows=[]
for rec in records:
    base=root/'data/external/dino_cohort'/Path(rec['filename']).stem;dest=base/'trellis_features.npz'
    if dest.exists():rows.append(json.load(open(base/'trellis_features.json')));continue
    data=np.load(root/'data/external/prepared'/rec['filename']);pr=np.load(base/'projection.npz');surface=np.unique(np.floor((data['points']+.5)*64).astype(int).clip(0,63),axis=0)
    # Projection at point positions; unlike round2, retain occluded-view features.
    patches=[]
    with torch.inference_mode():
        for i in range(4):
            image=np.asarray(Image.open(base/f'view_{i}.png').convert('RGB').resize((518,518),Image.Resampling.LANCZOS));x=torch.tensor(image.transpose(2,0,1).copy(),device='mps',dtype=torch.float32)[None]/255
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            feats=model.forward_features(x);patch=feats['x_prenorm'][:,1+model.num_register_tokens:];patches.append(patch.cpu())
    patch=torch.cat(patches);maps=patch.reshape(4,37,37,1024).permute(0,3,1,2);grid=torch.tensor(pr['pixels']/224*2-1).reshape(4,-1,1,2)
    point_features=torch.nn.functional.grid_sample(maps,grid,align_corners=False)[...,0].mean(0).T.numpy()
    sc,sf=pool_cells(np.floor((data['points']+.5)*64).astype(int).clip(0,63),point_features)
    bf=data['weights'].T@point_features/(data['weights'].sum(0)[:,None]+1e-8);rc,rf,_=rig_voxels(data['bones'][:,0],data['bones'][:,1],bf,64)
    # Auxiliary normalized patch mean; actual spatial flow conditions are cached separately.
    image_cond=torch.nn.functional.layer_norm(patch,(1024,)).mean(1).numpy()
    np.savez_compressed(dest,surface_coords=sc,surface_features=sf.astype(np.float16),rig_coords=rc,rig_features=rf.astype(np.float16),image_features=image_cond.astype('float32'))
    row=dict(**rec,trellis_feature_sha256=hashlib.sha256(dest.read_bytes()).hexdigest());rows.append(row);(base/'trellis_features.json').write_text(json.dumps(row,indent=2)+'\n');print(rec['filename'],sf.shape,flush=True)
weights=root/'data/external/torchhub/checkpoints/dinov2_vitl14_reg4_pretrain.pth'
report=dict(model='dinov2_vitl14_reg',channels=1024,feature='x_prenorm before final norm; register/CLS removed',input_size=518,native_render_size=224,views=4,aggregation='bilinear point projection, all-view mean then voxel pool; no visibility gating. Voxel centers are not directly projected, unlike original toolkit.',weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),source_commit=rev,scope='Real DINO-L features; GT rig/skin pooled skeleton inputs only in reconstruction/training. Native224 neutral views and point-to-voxel pooling differ from original renders.',assets=rows)
(root/'results/round3/dino_features.json').write_text(json.dumps(report,indent=2)+'\n')
