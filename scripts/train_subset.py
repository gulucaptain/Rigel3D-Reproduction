"""Multi-seed original-data decoder pilot; val selects checkpoints, test once."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from rigel3d.data import load_assets,fps_indices
from rigel3d.decoders import RigDecoders,skeleton_loss
from rigel3d.metrics import skeleton_metrics,skinning_metrics,point_segment_distance
from rigel3d.representation import decode_tokens
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
ap.add_argument('--mode',choices=['rig','geometry'],default='rig');ap.add_argument('--device',default='cpu');ap.add_argument('--name',default='round1')
ap.add_argument('--points',type=int,default=256);ap.add_argument('--batch',type=int,default=4);ap.add_argument('--width',type=int,default=64);ap.add_argument('--layers',type=int,default=2)
ap.add_argument('--balanced-stop',action='store_true')
ap.add_argument('--context',choices=['geometry','dino','vae','trellis'],default='geometry');ap.add_argument('--cohort',action='store_true');ap.add_argument('--vae-run',default='vae4000')
args=ap.parse_args();torch.set_num_threads(4)
if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
base_groups=load_assets(root);device=args.device
channels={'geometry':16,'dino':384,'vae':8,'trellis':8}[args.context]

def context_groups(seed):
    from rigel3d.experiment_data import load_context_assets
    return load_context_assets(root,dict(**vars(args),seed=seed))

def batch(rows,rng=None,eval_points=None):
    # Per-asset independent sample sets; fixed IDs at evaluation, random in train.
    count=eval_points or args.points;nb=max(len(r['bones']) for r in rows);nt=max(max(len(r['tokens']),1) for r in rows)
    ns=max(len(r['surface_coords']) for r in rows);nr=max(len(r['rig_coords']) for r in rows);b=len(rows)
    out={k:np.zeros(shape,dtype=np.float32) for k,shape in dict(points=(b,count,3),normals=(b,count,3),weights=(b,count,nb),heads=(b,nb,3),tails=(b,nb,3),sf=(b,ns,channels),sc=(b,ns,3),rf=(b,nr,channels),rc=(b,nr,3)).items()}
    out.update(tokens=np.zeros((b,nt,4),np.int64),lengths=np.zeros(b,np.int64),bone_pad=np.ones((b,nb),bool),surface_pad=np.ones((b,ns),bool),rig_pad=np.ones((b,nr),bool))
    ids=[]
    for i,r in enumerate(rows):
        valid=np.flatnonzero(r['skin_valid'])
        if len(valid)<count:raise ValueError('Too few weighted points')
        rr=rng or np.random.default_rng(int(hashlib.sha256(r['record']['asset_name'].encode()).hexdigest()[:8],16)+37)
        ix=rr.choice(valid,count,replace=False);ids.append(ix)
        out['points'][i]=r['points'][ix];out['normals'][i]=r['normals'][ix];out['weights'][i,:,:len(r['bones'])]=r['weights'][ix]
        out['heads'][i,:len(r['bones'])]=r['bones'][:,0];out['tails'][i,:len(r['bones'])]=r['bones'][:,1];out['bone_pad'][i,:len(r['bones'])]=False
        for pre,key in [('s','surface'),('r','rig')]:
            features=r[key+'_features'] if key=='surface' or args.mode=='rig' else r['geometry_rig_features']
            n=len(features);out[pre+'f'][i,:n]=features;out[pre+'c'][i,:n]=r[key+'_coords'];out[key+'_pad'][i,:n]=False
        if len(r['tokens']):out['tokens'][i,:len(r['tokens'])]=r['tokens'];out['lengths'][i]=len(r['tokens'])
    return {k:torch.as_tensor(v,device=device) for k,v in out.items()},ids


def run_model(model,d):
    s,r=model.encoder(d['sf'],d['sc'],d['rf'],d['rc'])
    w=model.skinning(d['points'],d['normals'],d['heads'],d['tails'],s,r,d['bone_pad'],d['surface_pad'],d['rig_pad'])
    skin=-(d['weights']*w.clamp_min(1e-12).log()).sum(-1).mean()
    eligible=d['lengths']>0
    if eligible.any():
        tokens=d['tokens'][eligible];lengths=d['lengths'][eligible];seqpad=torch.arange(tokens.shape[1]+1,device=device)[None]>lengths[:,None]
        out=model.skeleton(tokens,s[eligible],r[eligible],d['surface_pad'][eligible],d['rig_pad'][eligible],seqpad)
        skel,terms=skeleton_loss(out,tokens,lengths,balanced_stop=args.balanced_stop)
    else:skel=skin*0;terms={}
    return skin+skel,dict(skin=float(skin.detach().cpu()),skeleton=float(skel.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in terms.items()}),w


@torch.no_grad()
def validation(model):
    model.eval();vals=[]
    for row in groups['val']:
        d,_=batch([row]);loss,terms,_=run_model(model,d);vals.append(dict(total=float(loss.cpu()),**terms))
    return {k:float(np.mean([v[k] for v in vals if k in v])) for k in ['total','skin','skeleton','coords','parent','stop']}


@torch.no_grad()
def evaluate(model,rows):
    model.eval();result=[];predictions=[]
    for row in rows:
        d,ids=batch([row],eval_points=512);loss,terms,w=run_model(model,d);s,r=model.encoder(d['sf'],d['sc'],d['rf'],d['rc'])
        metrics=skinning_metrics(w[0,:,:len(row['bones'])].cpu().numpy(),row['weights'][ids[0]])
        rec=dict(asset_name=row['record']['asset_name'],skin=metrics,teacher_forced=terms,skeleton_eligible=bool(len(row['tokens'])))
        tokens,diag=model.skeleton.generate(s,r,d['surface_pad'],d['rig_pad']);tokens=tokens.cpu().numpy();j,p=decode_tokens(tokens)
        if len(row['tokens']):
            rec['skeleton']=dict(**skeleton_metrics(j,p,row['joints'],row['parents']),**diag)
        predictions.append(dict(tokens=tokens,weights=w[0,:,:len(row['bones'])].cpu().numpy(),point_ids=ids[0]))
        result.append(rec)
    return result,predictions


for seed in args.seeds:
    groups=context_groups(seed)
    folder=root/'results'/args.name/args.mode/f'seed{seed}'
    if folder.exists():raise FileExistsError(f'Choose another --name; preserving {folder}')
    folder.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=RigDecoders(args.width,args.layers,max_joints=96,channels=channels).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    config=dict(**vars(args),seed=seed,parameters=sum(p.numel() for p in model.parameters()),lr=1e-4,warmup_steps=min(200,args.steps//5),
        scheduler='linear warmup then cosine',source_manifest_sha256=hashlib.sha256((root/'results/anymate_preparation.json').read_bytes()).hexdigest(),
        task_scope=f'{args.context} context decoder training, cohort={args.cohort}. GT rig coordinates and skin weights in rig context; GT bones at skinning evaluation. Reconstruction, not image generation.',
        checkpoint_selection='Minimum validation teacher-forced skeleton+skin CE; test not used for selection')
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n');best=float('inf');start=time.perf_counter()
    with (folder/'trace.jsonl').open('w') as log:
        for step in range(args.steps):
            model.train();d,_=batch([groups['train'][i] for i in rng.choice(len(groups['train']),args.batch,replace=False)],rng)
            warm=config['warmup_steps'];factor=min((step+1)/max(warm,1),1.) if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(args.steps-warm,1)))
            for pg in opt.param_groups:pg['lr']=1e-4*factor
            opt.zero_grad(set_to_none=True);loss,terms,_=run_model(model,d)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss step {step}')
            loss.backward()
            grad_ok=torch.isfinite(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])).all()
            if not grad_ok:raise FloatingPointError(f'Nonfinite gradient step {step}')
            opt.step()
            if step%100==0 or step==args.steps-1:
                row=dict(step=step,loss=float(loss.detach().cpu()),elapsed_s=time.perf_counter()-start,**terms)
                if step%200==0 or step==args.steps-1:
                    row['validation']=validation(model)
                    if row['validation']['total']<best:
                        best=row['validation']['total'];best_step=step
                        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},folder/'best.pt')
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(dict(seed=seed,mode=args.mode,**row)),flush=True)
    model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True))
    evaluations={}
    for split,rows in [('train_monitor',groups['train'][:16]),('val',groups['val']),('test',groups['test'])]:
        records,preds=evaluate(model,rows);evaluations[split]=records
        # Only local ignored files carry pointwise weights/tokens, not public summary.
        dest=root/'data/external/predictions'/args.name/args.mode/f'seed{seed}'/split;dest.mkdir(parents=True,exist_ok=True)
        for i,pr in enumerate(preds):np.savez_compressed(dest/f'{i:03}.npz',**pr)
    (folder/'evaluation.json').write_text(json.dumps(dict(config=config,best_step=best_step,best_validation=best,elapsed_s=time.perf_counter()-start,splits=evaluations),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(seed=seed,mode=args.mode,complete=True,elapsed_s=time.perf_counter()-start)),flush=True)
