"""Paired512-point GT-bone skin audit; preserve earlier fixed-seed37 diagnostics."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from rigel3d.data import load_assets,fps_indices
from rigel3d.decoders import RigDecoders
from rigel3d.trellis_native import load_pretrained,Sparse
from rigel3d.metrics import skinning_metrics
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);allowed={r['filename'] for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets']};tests=[r for r in load_assets(root)['test'] if r['record']['filename'] in allowed];checks=[]
for run in ['joint_frozen_surface','full_joint']:
    for seed in [0,1,2]:
        folder=root/f'results/round3/{run}/seed{seed}'
        if run=='full_joint':manifest=json.load(open(folder/'model_manifest.json'));state=torch.load(root/manifest['checkpoint'],weights_only=True);encoder=load_pretrained('encoder',root/'data/external/trellis/weights','mps').float().eval();encoder.load_state_dict({k.removeprefix('rig_encoder.'):v for k,v in state.items() if k.startswith('rig_encoder.')},strict=True)
        else:state=torch.load(folder/'best.pt',weights_only=True);encoder=None
        dec=RigDecoders(channels=8).to('mps').eval();dec.load_state_dict({k.removeprefix('rig_decoders.'):v for k,v in state.items() if k.startswith('rig_decoders.')});heads={}
        if encoder is None:
            for mode in ['surface','rig']:
                h=torch.nn.Linear(768,16).to('mps');h.load_state_dict({k:state[mode+'_head.'+k] for k in ['weight','bias']});heads[mode]=h.eval()
        del state;records=[]
        with torch.no_grad():
            for row in tests:
                base=root/'data/external/dino_cohort'/Path(row['record']['filename']).stem;d=np.load(base/'trellis_latents.npz');lat={}
                for mode in ['surface','rig']:
                    ix=fps_indices(d[mode+'_coords'],48)
                    if encoder is not None:
                        if mode=='surface':mu=torch.tensor(d['surface_mu'][ix],device='mps')
                        else:
                            feat=np.load(base/'trellis_features.npz')['rig_features'].astype('float32');z,_,_=encoder(Sparse(torch.tensor(feat,device='mps'),d['rig_coords']),False);mu=z.feats[ix]
                    else:mu=heads[mode](torch.tensor(d[mode+'_hidden'][ix].astype('float32'),device='mps'))[:,:8]
                    lat[mode]=(mu[None],torch.tensor(((d[mode+'_coords'][ix]+.5)/64-.5).astype('float32'),device='mps')[None])
                s,r=dec.encoder(*lat['surface'],*lat['rig']);rng=np.random.default_rng(int(hashlib.sha256(row['record']['asset_name'].encode()).hexdigest()[:8],16)+37);ix=rng.choice(np.flatnonzero(row['skin_valid']),512,False)
                def t(v):return torch.tensor(v,device='mps')[None]
                w=dec.skinning(t(row['points'][ix]),t(row['normals'][ix]),t(row['bones'][:,0]),t(row['bones'][:,1]),s,r)[0].cpu().numpy();records.append(dict(filename=row['record']['filename'],point_ids_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),metrics=skinning_metrics(w,row['weights'][ix])))
        checks.append(dict(run=run,seed=seed,assets=records));del encoder,dec;torch.mps.empty_cache()
(root/'results/round3/paired_skin_audit.json').write_text(json.dumps(dict(scope='Same per-asset SHA-seeded512 weighted-point IDs as round3_frozen control. GT-bone reconstruction only. Earlier joint evaluation.json used a fixed37 RNG sample; those diagnostics are preserved, and comparison tables use this paired audit instead.',checks=checks),indent=2)+'\n');print('Paired skin audit complete')
