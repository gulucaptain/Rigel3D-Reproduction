"""Actual updated full rig-encoder means; no cached-old-trunk shortcut."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from rigel3d.joint_trellis import FullJointTrellis
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);records=json.load(open(root/'results/round3/pretrained_latents.json'))['assets'];rows=[]
for seed in [0,1,2]:
    folder=root/f'results/round3/full_joint/seed{seed}';assert (folder/'training.json').exists();manifest=json.load(open(folder/'model_manifest.json'));checkpoint=root/manifest['checkpoint'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==manifest['sha256'];model=FullJointTrellis(root/'data/external/trellis/weights').eval();model.load_state_dict(torch.load(checkpoint,weights_only=True),strict=True);dest=root/f'data/external/round3_full_joint_latents/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for r in records:
            base=root/'data/external/dino_cohort'/Path(r['filename']).stem;d=np.load(base/'trellis_latents.npz');feat=np.load(base/'trellis_features.npz')['rig_features'].astype('float32');z,mu,lv=model.encode_rig(torch.tensor(feat,device='mps'),d['rig_coords'],False);p=dest/r['filename'];np.savez_compressed(p,surface_coords=d['surface_coords'],surface_mu=d['surface_mu'],rig_coords=d['rig_coords'],rig_mu=mu.cpu().numpy(),rig_logvar=lv.cpu().numpy());rows.append(dict(seed=seed,filename=r['filename'],split=r['split'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    del model;torch.mps.empty_cache();print(seed,flush=True)
(root/'results/round3/full_joint_latents.json').write_text(json.dumps(dict(scope='Full updated rig transformer evaluated on real1024-channel GT-rig-pooled DINO features; surface original frozen means unchanged. Generator supervision targets only.',assets=rows),indent=2)+'\n')
