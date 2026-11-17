"""Three-seed coupled occupancy/local flows on original pretrained VAE latents."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from rigel3d.trellis_flow import CoupledFlows
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--balanced-rig-occupancy',action='store_true');ap.add_argument('--joint-run',default='joint_frozen_surface');ap.add_argument('--name',default='flows_frozen_surface');a=ap.parse_args();torch.set_num_threads(4);records=json.load(open(root/'results/round3/pretrained_latents.json'))['assets'];groups={s:[r for r in records if r['split']==s] for s in ['train','val','test']}
for seed in a.seeds:
    folder=root/f'results/round3/{a.name}/seed{seed}';folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);rng=np.random.default_rng(seed);stats={};cache={}
    for r in records:
        base=root/'data/external/dino_cohort'/Path(r['filename']).stem;occ=dict(np.load(base/'occupancy_pretrained.npz'));lat=dict(np.load(root/f'data/external/round3_{a.joint_run}_latents/seed{seed}'/r['filename']));image=np.load(base/'trellis_image_tokens.npz')['tokens'].astype('float32');cache[r['filename']]=dict(occ=occ,lat=lat,image=image)
    for mode in ['surface','rig']:
        grids=np.stack([cache[r['filename']]['occ'][mode+'_mu'] for r in groups['train']]);mu=grids.mean((0,2,3,4));std=grids.std((0,2,3,4)).clip(.1);stats[mode+'_occ_mean']=mu;stats[mode+'_occ_std']=std
        z=np.concatenate([cache[r['filename']]['lat'][mode+'_mu'] for r in groups['train']]);stats[mode+'_local_mean']=z.mean(0);stats[mode+'_local_std']=z.std(0).clip(.1)
    data={}
    for r in records:
        d=cache[r['filename']];item=dict(image=torch.tensor(d['image'],device='mps'));grids=[]
        for mode in ['surface','rig']:
            g=(d['occ'][mode+'_mu']-stats[mode+'_occ_mean'][:,None,None,None])/stats[mode+'_occ_std'][:,None,None,None];grids.append(g)
            item[mode]=dict(coords=torch.tensor(((d['lat'][mode+'_coords']+.5)/64-.5).astype('float32'),device='mps'),z=torch.tensor((d['lat'][mode+'_mu']-stats[mode+'_local_mean'])/stats[mode+'_local_std'],device='mps'))
        item['occ']=torch.tensor(np.concatenate(grids)[None],device='mps')
        support=np.zeros((64,64,64),np.float32);c=d['lat']['rig_coords'];support[c[:,0],c[:,1],c[:,2]]=1;item['rig_support']=torch.nn.functional.max_pool3d(torch.tensor(support[None,None]),4).bool()[0,0].to('mps')
        data[r['filename']]=item
    model=CoupledFlows().to('mps');opt=torch.optim.AdamW(model.parameters(),lr=1e-4);best=float('inf');trace=[];start=time.perf_counter()
    def objective(item,validation=False):
        image=item['image'][0:1] if validation else item['image'][int(rng.integers(4)):][:1]
        if not validation and rng.random()<.1:image=torch.zeros_like(image)
        x1=item['occ'];x0=torch.randn_like(x1);t=torch.rand(1,device='mps');xt=(1-t)*x0+t*x1;velocity=model.velocity(xt,t,image);err=(velocity-(x1-x0)).square();occ_loss=err.mean()
        if a.balanced_rig_occupancy:
            rig_err=err[:,8:].mean((0,1));mask=item['rig_support'];occ_loss=.5*err[:,:8].mean()+.25*rig_err[mask].mean()+.25*rig_err[~mask].mean()
        local=occ_loss*0
        for mode in ['surface','rig']:
            d=item[mode];ix=torch.randperm(len(d['z']),device='mps')[:128];z1=d['z'][ix];z0=torch.randn_like(z1);zt=(1-t)*z0+t*z1;local=local+(model.local[mode](zt,t,image,d['coords'][ix],x1)-(z1-z0)).square().mean()
        return occ_loss+local,occ_loss,local
    for step in range(a.steps):
        model.train();item=data[groups['train'][int(rng.integers(len(groups['train'])))]['filename']];opt.zero_grad();loss,ol,ll=objective(item)
        if not torch.isfinite(loss):raise FloatingPointError('flow loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0 or step==a.steps-1:
            rec=dict(step=step,loss=float(loss.detach().cpu()),occupancy=float(ol.detach().cpu()),local=float(ll.detach().cpu()),elapsed_s=time.perf_counter()-start)
            if step%200==0 or step==a.steps-1:
                model.eval();rng_state=torch.mps.get_rng_state()
                with torch.no_grad(),torch.random.fork_rng(devices=[]):
                    torch.manual_seed(1234);val=float(torch.stack([objective(data[r['filename']],True)[0] for r in groups['val']]).mean().cpu())
                torch.mps.set_rng_state(rng_state);rec['validation']=val
                if val<best:best=val;best_step=step;torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'best.pt')
            trace.append(rec);print(seed,json.dumps(rec),flush=True)
    np.savez(folder/'stats.npz',**stats);(folder/'training.json').write_text(json.dumps(dict(seed=seed,joint_run=a.joint_run,balanced_rig_occupancy=a.balanced_rig_occupancy,lr=1e-4,steps=a.steps,best_step=best_step,best_validation=best,parameters=sum(p.numel() for p in model.parameters()),trace=trace,scope='Coupled Mac-size3D conv occupancy flows over full16^3x8 official-VAE latents; spatial image-token CA, two local sparse flows. Architecture smaller than original transformers. GT states are supervised targets only; sampling uses image/noise and predicted occupied64^3 positions.'),indent=2)+'\n');del model,opt;torch.mps.empty_cache()
