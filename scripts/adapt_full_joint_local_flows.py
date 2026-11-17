"""Align local flows to updated full rig encoder; keep occupancy generators fixed."""
import json,hashlib,shutil,time
from pathlib import Path
import numpy as np,torch
from rigel3d.trellis_flow import CoupledFlows
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);records=json.load(open(root/'results/round3/pretrained_latents.json'))['assets'];groups={s:[r for r in records if r['split']==s] for s in ['train','val','test']}
for source,destination in [('flows_frozen_surface','flows_full_joint'),('flows_balanced_rig','flows_full_joint_balanced')]:
    for seed in [0,1,2]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);src=root/f'results/round3/{source}/seed{seed}';dest=root/f'results/round3/{destination}/seed{seed}';dest.mkdir(parents=True,exist_ok=False);stats=dict(np.load(src/'stats.npz'));data={}
        for r in records:
            base=root/'data/external/dino_cohort'/Path(r['filename']).stem;occ=np.load(base/'occupancy_pretrained.npz');lat=np.load(root/f'data/external/round3_full_joint_latents/seed{seed}'/r['filename']);item=dict(image=torch.tensor(np.load(base/'trellis_image_tokens.npz')['tokens'].astype('float32'),device='mps'));grids=[]
            for mode in ['surface','rig']:
                grids.append((occ[mode+'_mu']-stats[mode+'_occ_mean'][:,None,None,None])/stats[mode+'_occ_std'][:,None,None,None]);item[mode]=dict(z=torch.tensor((lat[mode+'_mu']-stats[mode+'_local_mean'])/stats[mode+'_local_std'],device='mps'),coords=torch.tensor(((lat[mode+'_coords']+.5)/64-.5).astype('float32'),device='mps'))
            item['occ']=torch.tensor(np.concatenate(grids)[None],device='mps');data[r['filename']]=item
        model=CoupledFlows().to('mps');original=torch.load(src/'best.pt',weights_only=True);model.load_state_dict(original);model.occupancy.requires_grad_(False);opt=torch.optim.AdamW(model.local.parameters(),lr=1e-4,weight_decay=0);trace=[];best=float('inf');start=time.perf_counter()
        def loss_for(item,validation=False):
            image=item['image'][:1] if validation else item['image'][int(rng.integers(4)):][:1]
            if not validation and rng.random()<.1:image=torch.zeros_like(image)
            loss=0
            for mode in ['surface','rig']:
                d=item[mode];ix=torch.randperm(len(d['z']),device='mps')[:128];z1=d['z'][ix];z0=torch.randn_like(z1);t=torch.rand(1,device='mps');zt=(1-t)*z0+t*z1;loss=loss+(model.local[mode](zt,t,image,d['coords'][ix],item['occ'])-(z1-z0)).square().mean()
            return loss
        for step in range(1000):
            model.train();opt.zero_grad();item=data[groups['train'][int(rng.integers(len(groups['train'])))]['filename']];loss=loss_for(item)
            if not torch.isfinite(loss):raise FloatingPointError('local re-alignment loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.local.parameters(),1)
            if not torch.isfinite(norm):raise FloatingPointError('local gradients')
            opt.step()
            if step%200==0 or step==999:
                model.eval();mps_rng=torch.mps.get_rng_state()
                with torch.no_grad(),torch.random.fork_rng(devices=[]):
                    torch.manual_seed(1234);val=float(torch.stack([loss_for(data[r['filename']],True) for r in groups['val']]).mean().cpu())
                torch.mps.set_rng_state(mps_rng);trace.append(dict(step=step,loss=float(loss.detach().cpu()),validation=val,elapsed_s=time.perf_counter()-start))
                if val<best:best=val;best_step=step;torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},dest/'best.pt')
        saved=torch.load(dest/'best.pt',weights_only=True);assert all(torch.equal(v,saved[k]) for k,v in original.items() if k.startswith('occupancy.'));shutil.copyfile(src/'stats.npz',dest/'stats.npz');(dest/'training.json').write_text(json.dumps(dict(seed=seed,source=source,steps=1000,best_step=best_step,best_validation=best,source_sha256=hashlib.sha256((src/'best.pt').read_bytes()).hexdigest(),occupancy_bitwise_unchanged=True,stats_bitwise_unchanged=True,trace=trace,scope='Local latent flow re-alignment after full rig encoder update; original train-only scaling retained. Occupancy weights unchanged, so this stage cannot fix occupancy failures. No test selection.'),indent=2)+'\n');print(source,seed,'complete',flush=True);del model,opt;torch.mps.empty_cache()
