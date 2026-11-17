"""Warm-started short full-weight joint stage, three seeds; surface remains frozen."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.data import load_assets,fps_indices
from rigel3d.joint_trellis import FullJointTrellis
from rigel3d.trellis_mesh import cube_fields
from rigel3d.decoders import skeleton_loss
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=250);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);a=ap.parse_args();torch.set_num_threads(4);allowed={r['filename'] for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets']};groups={s:[r for r in rows if r['record']['filename'] in allowed] for s,rows in load_assets(root).items()};cache={}
for rows in groups.values():
    for row in rows:
        name=row['record']['filename'];base=root/'data/external/dino_cohort'/Path(name).stem;d=dict(np.load(base/'trellis_latents.npz'));d['rig_features']=np.load(base/'trellis_features.npz')['rig_features'].astype('float32');d['si']=fps_indices(d['surface_coords'],48);d['ri']=fps_indices(d['rig_coords'],48);d['tree']=cKDTree(row['points']);cache[name]=d
for seed in a.seeds:
    folder=root/f'results/round3/full_joint/seed{seed}';folder.mkdir(parents=True,exist_ok=False);weights_folder=root/f'data/external/full_joint/seed{seed}';weights_folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);rng=np.random.default_rng(seed);warm=root/f'results/round3/joint_frozen_surface/seed{seed}/best.pt';model=FullJointTrellis(root/'data/external/trellis/weights',torch.load(warm,weights_only=True));parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0);best=float('inf');trace=[];start=time.perf_counter()
    def objective(row,training=True,geometry=False):
        d=cache[row['record']['filename']];device='mps';surface=torch.tensor(d['surface_mu'],device=device);rz,mu,lv=model.encode_rig(torch.tensor(d['rig_features'],device=device),d['rig_coords'],training);kl=.5*(mu.square()+lv.exp()-lv-1).mean();anchor=(mu-torch.tensor(d['rig_mu'],device=device)).square().mean()
        sc=torch.tensor(((d['surface_coords']+.5)/64-.5).astype('float32'),device=device);rc=torch.tensor(((d['rig_coords']+.5)/64-.5).astype('float32'),device=device);s,r=model.rig_decoders.encoder(surface[d['si']][None],sc[d['si']][None],rz.feats[d['ri']][None],rc[d['ri']][None]);ix=(rng if training else np.random.default_rng(37)).choice(np.flatnonzero(row['skin_valid']),128,False)
        def t(v):return torch.tensor(v,device=device)[None]
        w=model.rig_decoders.skinning(t(row['points'][ix]),t(row['normals'][ix]),t(row['bones'][:,0]),t(row['bones'][:,1]),s,r);skin=-(t(row['weights'][ix])*w.clamp_min(1e-12).log()).sum(-1).mean();skel=skin*0
        if len(row['tokens']):
            target=t(row['tokens']);out=model.rig_decoders.skeleton(target,s,r);skel,_=skeleton_loss(out,target,torch.tensor([len(row['tokens'])],device=device),True)
        geom=skin*0
        if geometry:
            coords=d['surface_coords'];centre=coords[int(rng.integers(len(coords)))] if training else coords[0];patch=np.argsort(np.square(coords-centre).sum(-1),kind='stable')[:16];fields=model.mesh_fields(surface[patch],coords[patch],rz.feats[d['ri']],rc[d['ri']]);vp,attrs,cons=cube_fields(fields);q=vp.float()/256-.5+attrs[:,1:4].tanh()/512;nearest=d['tree'].query(q.detach().cpu().numpy())[1];p=torch.tensor(row['points'][nearest],device=device);normal=torch.tensor(row['normals'][nearest],device=device);delta=q-p;sign=torch.where((delta.detach()*normal).sum(-1)>=0,1.,-1.);distance=delta.norm(dim=-1)*sign;geom=(attrs[:,0]-distance.clamp(-.15,.15)).abs().mean()/.15+.1*cons
        return skin+skel+geom+1e-6*kl+.01*anchor,dict(skin=float(skin.detach().cpu()),skeleton=float(skel.detach().cpu()),geometry=float(geom.detach().cpu()),kl=float(kl.detach().cpu()),anchor=float(anchor.detach().cpu()))
    @torch.no_grad()
    def validation():
        model.eval();return float(np.mean([float(objective(row,False,True)[0].cpu()) for row in groups['val']]))
    for step in range(a.steps):
        model.train();warmup=min(25,a.steps//5);factor=min((step+1)/max(warmup,1),1) if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(a.steps-warmup,1)))
        for pg in opt.param_groups:pg['lr']=1e-4*factor
        opt.zero_grad();row=groups['train'][int(rng.integers(len(groups['train'])))];loss,terms=objective(row,True,step%8==0)
        if not torch.isfinite(loss):raise FloatingPointError('full joint loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1)
        if not torch.isfinite(norm):raise FloatingPointError('full joint gradients')
        opt.step()
        if step%25==0 or step==a.steps-1:
            rec=dict(step=step,loss=float(loss.detach().cpu()),elapsed_s=time.perf_counter()-start,allocated_gb=torch.mps.current_allocated_memory()/1e9,**terms)
            if step%50==0 or step==a.steps-1:
                rec['validation']=validation()
                if rec['validation']<best:
                    best=rec['validation'];best_step=step;torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},weights_folder/'best.pt')
            trace.append(rec);print(seed,json.dumps(rec),flush=True)
    checkpoint=weights_folder/'best.pt';manifest=dict(seed=seed,checkpoint=str(checkpoint.relative_to(root)),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),bytes=checkpoint.stat().st_size);(folder/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(folder/'training.json').write_text(json.dumps(dict(seed=seed,steps=a.steps,best_step=best_step,best_validation=best,trainable_parameters=sum(p.numel() for p in parameters),warm_start_sha256=hashlib.sha256(warm.read_bytes()).hexdigest(),lr=1e-4,weight_decay=0,warmup=25,geometry_every=8,patch_cubes=16,trace=trace,scope='After2000-step adapter warm start, full original85.8M rig encoder +90.9M mesh weights + rig CA/AR/skin jointly updated; surface encoder remains frozen. Only250 short steps, small-patch oriented-SDF surrogate, not full rendering-loss convergence.'),indent=2)+'\n');del model,opt;torch.mps.empty_cache()
