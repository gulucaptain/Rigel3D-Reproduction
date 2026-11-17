"""CPU replay of frozen VAE posterior means and test losses; verify source hashes."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from rigel3d.sparse_vae import SparseVAE,neighbor_indices
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);records=json.load(open(root/'results/round2/dino_features.json'))['assets'];checks=[]
for row in records:
    p=root/'data/external/dino_cohort'/Path(row['filename']).stem/'features.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
for seed in [0,1,2]:
    for mode in ['surface','rig']:
        folder=root/f'results/round2/vae4000/seed{seed}/{mode}';report=json.load(open(folder/'evaluation.json'));norm=np.load(folder/'normalization.npz');model=SparseVAE().eval();model.load_state_dict(torch.load(folder/'best.pt',weights_only=True));losses=[];maximum=0;latents=[]
        with torch.no_grad():
            for row in records:
                p=root/'data/external/dino_cohort'/Path(row['filename']).stem/'features.npz';d=np.load(p);x=torch.tensor((d[mode+'_features']-norm['mean'])/norm['std']);n=torch.tensor(neighbor_indices(d[mode+'_coords']));mu,_=model.encode(x,n);lp=root/f'data/external/vae4000_latents/seed{seed}'/(mode+'_'+row['filename']);saved=np.load(lp);err=float(np.abs(mu.numpy()-saved['features']).max());maximum=max(maximum,err)
                assert err<2e-4,(seed,mode,row['filename'],err)
                if row['split']=='test':losses.append(float((model.decode(mu,n)-x).square().mean()))
                latents.append(dict(filename=row['filename'],sha256=hashlib.sha256(lp.read_bytes()).hexdigest()))
        mse=float(np.mean(losses));delta=abs(mse-report['metrics']['test']['mse']);assert delta<2e-4,(seed,mode,delta)
        checks.append(dict(seed=seed,mode=mode,passed=True,posterior_max_abs_difference=maximum,test_mse_cpu=mse,test_mse_difference=delta,checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),normalization_sha256=hashlib.sha256((folder/'normalization.npz').read_bytes()).hexdigest(),latent_files=latents))
report=dict(passed=True,source_feature_hashes_checked=len(records),checks=checks);(root/'results/round2/vae_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,models=len(checks))))
