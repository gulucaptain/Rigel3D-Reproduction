"""Train-only image flow/field pilot. Validation chooses each checkpoint; test held out."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from rigel3d.generation import FourFlows,FieldDecoder
from rigel3d.data import fps_indices
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=ap.parse_args();torch.set_num_threads(4)
records=json.loads((root/'results/round2/dino_features.json').read_text())['assets'];groups={s:[r for r in records if r['split']==s] for s in ['train','val','test']}
for seed in a.seeds:
    torch.manual_seed(seed);rng=np.random.default_rng(seed);folder=root/f'results/round2/generation/seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    data={};stats={}
    for mode in ['surface','rig']:
        lat=np.concatenate([np.load(root/f'data/external/vae4000_latents/seed{seed}'/(mode+'_'+r['filename']))['features'] for r in groups['train']]);stats[mode]=(lat.mean(0),lat.std(0).clip(.1))
    for r in records:
        name=r['filename'];base=root/'data/external/dino_cohort'/Path(name).stem;d=np.load(base/'features.npz');sd=np.load(base/'sdf.npz');item=dict(image=torch.tensor(d['image_features'].mean(1),device='mps'),query=torch.tensor(sd['points'],device='mps'),sdf=torch.tensor(sd['sdf'],device='mps'))
        for mode in ['surface','rig']:
            l=np.load(root/f'data/external/vae4000_latents/seed{seed}'/(mode+'_'+name));c=l['coords'];f=l['features'];ix=fps_indices(c,48);occ=np.full(16**3,-1,np.float32);coarse=np.clip(c//4,0,15);occ[coarse[:,0]*256+coarse[:,1]*16+coarse[:,2]]=1
            mu,std=stats[mode];item[mode]=dict(occ=torch.tensor(occ[None],device='mps'),coords=torch.tensor(((c+.5)/64-.5).astype('float32'),device='mps'),latent=torch.tensor((f-mu)/std,device='mps'),f=torch.tensor(f[ix][None],device='mps'),c=torch.tensor(((c[ix]+.5)/64-.5).astype('float32')[None],device='mps'))
        data[name]=item
    flows=FourFlows().to('mps');field=FieldDecoder().to('mps');opt=torch.optim.AdamW(list(flows.parameters())+list(field.parameters()),lr=3e-4);trace=[];best_flow=float('inf');best_field=float('inf');start=time.perf_counter()
    def objective(item,deterministic=False):
        image=item['image'][0:1] if deterministic else item['image'][int(rng.integers(4)):][:1]
        if not deterministic and rng.random()<.1:image=torch.zeros_like(image)
        losses=[]
        for mode in ['surface','rig']:
            d=item[mode];x1=d['occ'];x0=torch.randn_like(x1);t=torch.rand(1,device='mps');xt=(1-t)*x0+t*x1;v=flows.occupancy[mode](xt,t,image)
            weights=torch.where(x1>0,8.,1.);losses.append(((v-(x1-x0)).square()*weights).mean()/weights.mean())
            ix=torch.randperm(len(d['coords']),device='mps')[:128];z1=d['latent'][ix];z0=torch.randn_like(z1);zt=(1-t)*z0+t*z1
            losses.append((flows.local[mode](zt,t,image,d['coords'][ix])-(z1-z0)).square().mean())
        q=torch.randperm(len(item['query']),device='mps')[:512];pred=field(item['query'][q][None],item['surface']['f'],item['surface']['c'],item['rig']['f'],item['rig']['c'])[0];target=item['sdf'][q]
        field_loss=(pred-target).abs().mean()/.15+.1*torch.nn.functional.binary_cross_entropy_with_logits(-pred/.02,(target<0).float())
        return sum(losses),field_loss
    for step in range(a.steps):
        flows.train();field.train();item=data[groups['train'][int(rng.integers(len(groups['train'])))]['filename']];opt.zero_grad();lf,lg=objective(item);loss=lf+lg
        if not torch.isfinite(loss):raise FloatingPointError('Flow/field loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(list(flows.parameters())+list(field.parameters()),1);opt.step()
        if step%200==0 or step==a.steps-1:
            flows.eval();field.eval()
            # Fixed forked torch RNG and no mutation of NumPy generator for validation.
            mps_rng=torch.mps.get_rng_state()
            with torch.no_grad(),torch.random.fork_rng(devices=[]):
                torch.manual_seed(1234);vals=[objective(data[r['filename']],True) for r in groups['val']];vf=float(torch.stack([v[0] for v in vals]).mean().cpu());vg=float(torch.stack([v[1] for v in vals]).mean().cpu())
            torch.mps.set_rng_state(mps_rng)
            rec=dict(step=step,flow=float(lf.detach().cpu()),field=float(lg.detach().cpu()),val_flow=vf,val_field=vg,elapsed_s=time.perf_counter()-start);trace.append(rec);print(seed,json.dumps(rec),flush=True)
            for model,value,key in [(flows,vf,'flows'),(field,vg,'field')]:
                best=best_flow if key=='flows' else best_field
                if value<best:
                    torch.save({k:v.cpu().clone() for k,v in model.state_dict().items()},folder/(key+'.pt'))
                    if key=='flows':best_flow=value
                    else:best_field=value
    np.savez(folder/'latent_stats.npz',**{m+'_'+k:v for m,(mu,std) in stats.items() for k,v in [('mean',mu),('std',std)]})
    (folder/'training.json').write_text(json.dumps(dict(seed=seed,steps=a.steps,trace=trace,scope='Small independent image-flow and GT-latent field reconstruction training. Generation evaluated separately.',deviations=['DINOv2-S/224 neutral views','direct 16^3 occupancy; no occupancy VAE','independent MLP rectified flows; no flow cross-branch attention','nearest-normal weak SDF supervision; no original rendering losses','16-grid FlexiCubes; default blend weights; no learned deformation']),indent=2)+'\n')
