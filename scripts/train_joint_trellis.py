"""Three-seed joint head/adapter pilot with GT-conditioned validation only."""
import argparse,json,time,hashlib,math
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.data import load_assets,fps_indices
from rigel3d.joint_trellis import JointTrellis
from rigel3d.trellis_mesh import cube_fields
from rigel3d.decoders import skeleton_loss
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);ap.add_argument('--mesh-every',type=int,default=8);ap.add_argument('--patch-cubes',type=int,default=16);ap.add_argument('--name',default='joint_frozen_surface');ap.add_argument('--train-surface',action='store_true');a=ap.parse_args();torch.set_num_threads(4)
manifest=json.load(open(root/'results/round3/pretrained_latents.json'));allowed={r['filename'] for r in manifest['assets']};groups={s:[r for r in rs if r['record']['filename'] in allowed] for s,rs in load_assets(root).items()};cache={}
for rows in groups.values():
    for row in rows:
        name=row['record']['filename'];base=root/'data/external/dino_cohort'/Path(name).stem;d=dict(np.load(base/'trellis_latents.npz'));d['sidx']=fps_indices(d['surface_coords'],48);d['ridx']=fps_indices(d['rig_coords'],48);d['tree']=cKDTree(row['points']);cache[name]=d
for seed in a.seeds:
    folder=root/f'results/round3/{a.name}/seed{seed}';folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed);rng=np.random.default_rng(seed);model=JointTrellis(root/'data/external/trellis/weights');model.surface_head.requires_grad_(a.train_surface);parameters=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(parameters,lr=1e-4);start=time.perf_counter();trace=[];best=float('inf')
    def objective(row,training=True,mesh=False):
        d=cache[row['record']['filename']];device='mps';post={};kl=0;anchor=0
        for mode in ['surface','rig']:
            hidden=torch.tensor(d[mode+'_hidden'].astype('float32'),device=device);z,mu,lv,k=model.posterior(hidden,mode,training);post[mode]=(z,mu,lv);kl=kl+k;anchor=anchor+(mu-torch.tensor(d[mode+'_mu'],device=device)).square().mean()
        sf=post['surface'][0][d['sidx']][None];rf=post['rig'][0][d['ridx']][None];sc=torch.tensor(((d['surface_coords'][d['sidx']]+.5)/64-.5).astype('float32'),device=device);rc=torch.tensor(((d['rig_coords'][d['ridx']]+.5)/64-.5).astype('float32'),device=device)
        s,r=model.rig_decoders.encoder(sf,sc[None],rf,rc[None]);ix=(rng if training else np.random.default_rng(37)).choice(np.flatnonzero(row['skin_valid']),128,replace=False)
        def t(v):return torch.tensor(v,device=device)[None]
        w=model.rig_decoders.skinning(t(row['points'][ix]),t(row['normals'][ix]),t(row['bones'][:,0]),t(row['bones'][:,1]),s,r);skin=-(t(row['weights'][ix])*w.clamp_min(1e-12).log()).sum(-1).mean();skel=skin*0
        if len(row['tokens']):
            target=t(row['tokens']);out=model.rig_decoders.skeleton(target,s,r);skel,_=skeleton_loss(out,target,torch.tensor([len(row['tokens'])],device=device),True)
        geometry=skin*0
        if mesh:
            coords=d['surface_coords'];centre=coords[int(rng.integers(len(coords)))] if training else coords[0];dist=np.square(coords-centre).sum(-1);patch=np.argsort(dist,kind='stable')[:a.patch_cubes]
            fields=model.mesh_fields(post['surface'][0][patch],coords[patch],post['rig'][0][d['ridx']],rc)
            vp,attrs,cons=cube_fields(fields);q=vp.float()/256-.5+attrs[:,1:4].tanh()/512;nearest=d['tree'].query(q.detach().cpu().numpy())[1];target_p=torch.tensor(row['points'][nearest],device=device);target_n=torch.tensor(row['normals'][nearest],device=device);delta=q-target_p;sign=torch.where((delta.detach()*target_n).sum(-1)>=0,1.,-1.);signed=delta.norm(dim=-1)*sign
            geometry=(attrs[:,0]-signed.clamp(-.15,.15)).abs().mean()/.15+.1*cons
        loss=skin+skel+geometry+1e-6*kl+.01*anchor
        return loss,dict(skin=float(skin.detach().cpu()),skeleton=float(skel.detach().cpu()),geometry=float(geometry.detach().cpu()),kl=float(kl.detach().cpu()),posterior_anchor=float(anchor.detach().cpu()))
    @torch.no_grad()
    def validate():
        model.eval();items=[objective(r,False,True) for r in groups['val']];return float(np.mean([float(v[0].cpu()) for v in items]))
    for step in range(a.steps):
        model.train();warm=min(200,a.steps//5);factor=min((step+1)/max(warm,1),1) if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(a.steps-warm,1)))
        for pg in opt.param_groups:pg['lr']=1e-4*factor
        row=groups['train'][int(rng.integers(len(groups['train'])))];opt.zero_grad();loss,terms=objective(row,True,step%a.mesh_every==0)
        if not torch.isfinite(loss):raise FloatingPointError('joint loss '+str(step))
        loss.backward();grad=torch.cat([p.grad.flatten() for p in parameters if p.grad is not None]);
        if not torch.isfinite(grad).all():raise FloatingPointError('joint gradient '+str(step))
        torch.nn.utils.clip_grad_norm_(parameters,1);opt.step()
        if step%100==0 or step==a.steps-1:
            rec=dict(step=step,loss=float(loss.detach().cpu()),elapsed_s=time.perf_counter()-start,**terms)
            if step%200==0 or step==a.steps-1:
                rec['validation']=validate()
                if rec['validation']<best:best=rec['validation'];best_step=step;torch.save(model.trainable_state(),folder/'best.pt')
            trace.append(rec);print(seed,json.dumps(rec),flush=True)
    config=dict(**vars(a),seed=seed,trainable_parameters=sum(p.numel() for p in parameters),frozen_mesh_parameters=sum(p.numel() for p in model.mesh.parameters()),device='mps',selection='Minimum GT-conditioned validation joint loss; no test selection',scope='Joint rig-posterior/rig-CA/AR/skin adaptation; surface head frozen unless --train-surface. Cosine LR with scaled200 warmup (paper5000). frozen original encoder trunk via cached768 features and frozen mesh base. Mesh loss16-cube patches every8 steps, nearest oriented point distance + consistency; KL1e-6 and posterior anchor. Not full-backbone original rendering-loss training.',best_step=best_step,best_validation=best,trace=trace,source_manifest_sha256=hashlib.sha256((root/'results/round3/pretrained_latents.json').read_bytes()).hexdigest())
    (folder/'training.json').write_text(json.dumps(config,indent=2)+'\n');del model,opt;torch.mps.empty_cache()
