"""Three-seed surface+rig Sparse VAE with train-only channel statistics."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from rigel3d.sparse_vae import SparseVAE,neighbor_indices
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=600);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--name',default='vae');args=ap.parse_args();torch.set_num_threads(4)
report=json.loads((root/'results/round2/dino_features.json').read_text());records=report['assets'];cache={}
for row in records:
    f=root/'data/external/dino_cohort'/Path(row['filename']).stem;d=dict(np.load(f/'features.npz'))
    for mode in ['surface','rig']:
        d[mode+'_neighbors']=neighbor_indices(d[mode+'_coords'])
    cache[row['filename']]=d
for seed in args.seeds:
    for mode in ['surface','rig']:
        folder=root/f'results/round2/{args.name}/seed{seed}/{mode}';folder.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(seed);rng=np.random.default_rng(seed);rows={s:[r for r in records if r['split']==s] for s in ['train','val','test']}
        train=np.concatenate([cache[r['filename']][mode+'_features'] for r in rows['train']]);mean=train.mean(0);std=train.std(0).clip(.1)
        data={}
        for r in records:
            d=cache[r['filename']];data[r['filename']]=(torch.tensor((d[mode+'_features']-mean)/std,device='mps'),torch.tensor(d[mode+'_neighbors'],device='mps'))
        model=SparseVAE().to('mps');opt=torch.optim.AdamW(model.parameters(),lr=3e-4);best=float('inf');trace=[];start=time.perf_counter()
        @torch.no_grad()
        def evaluate(split):
            model.eval();out=[]
            for r in rows[split]:
                x,n=data[r['filename']];pred,kl,_=model(x,n,False);out.append(dict(mse=float((pred-x).square().mean().cpu()),kl=float(kl.cpu())))
            return {k:float(np.mean([a[k] for a in out])) for k in ['mse','kl']}
        for step in range(args.steps):
            model.train();r=rows['train'][int(rng.integers(len(rows['train'])))];x,n=data[r['filename']];opt.zero_grad();pred,kl,_=model(x,n);mse=(pred-x).square().mean();loss=mse+1e-4*kl
            if not torch.isfinite(loss):raise FloatingPointError('VAE loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step%100==0 or step==args.steps-1:
                val=evaluate('val');rec=dict(step=step,train_mse=float(mse.detach().cpu()),val=val,elapsed_s=time.perf_counter()-start);trace.append(rec);print(seed,mode,json.dumps(rec),flush=True)
                if val['mse']<best:
                    best=val['mse'];best_step=step;torch.save({k:v.cpu().clone() for k,v in model.state_dict().items()},folder/'best.pt')
        model.load_state_dict(torch.load(folder/'best.pt',weights_only=True));metrics={s:evaluate(s) for s in rows}
        # Deterministic posterior means for downstream reconstruction. Coordinates remain GT.
        with torch.no_grad():
            for r in records:
                x,n=data[r['filename']];mu,_=model.encode(x,n);dest=root/f'data/external/{args.name}_latents/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/(mode+'_'+r['filename']),coords=cache[r['filename']][mode+'_coords'],features=mu.cpu().numpy())
        np.savez(folder/'normalization.npz',mean=mean,std=std)
        result=dict(seed=seed,mode=mode,steps=args.steps,best_step=best_step,metrics=metrics,trace=trace,device='mps',channels=384,width=32,latent=8,kl_weight=1e-4,parameters=sum(p.numel() for p in model.parameters()),source_sha256=hashlib.sha256((root/'results/round2/dino_features.json').read_bytes()).hexdigest(),scope='GT occupied sparse coordinates; real DINO feature reconstruction. Not mesh or image generation.')
        (folder/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
