"""Separate GT-latent reconstruction from strict image-only generation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.data import load_assets,fps_indices
from rigel3d.decoders import RigDecoders
from rigel3d.generation import FourFlows,FieldDecoder
from rigel3d.flexicubes import regular_grid,FlexiCubes
from rigel3d.inference import generate_asset
from rigel3d.metrics import rignet_metrics
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--cfg',type=float,default=5);ap.add_argument('--output-name',default='generation');args=ap.parse_args();torch.set_num_threads(4)
records=[r for r in json.loads((root/'results/round2/dino_features.json').read_text())['assets'] if r['split']=='test'];targets={r['record']['filename']:r for r in load_assets(root)['test']}
def mesh_metric(v,f,target):
    if not len(f):return None
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)
    if area.sum()<1e-10:return None
    rng=np.random.default_rng(777);idx=rng.choice(len(f),2048,p=area/area.sum());uv=rng.random((2048,2));uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1];p=tri[idx,0]+uv[:,:1]*(tri[idx,1]-tri[idx,0])+uv[:,1:]*(tri[idx,2]-tri[idx,0]);return float((cKDTree(p).query(target)[0].mean()+cKDTree(target).query(p)[0].mean())/2)
for seed in [0,1,2]:
    folder=root/f'results/round2/generation/seed{seed}';flows=FourFlows().to('mps').eval();flows.load_state_dict(torch.load(folder/'flows.pt',weights_only=True));field=FieldDecoder().to('mps').eval();field.load_state_dict(torch.load(folder/'field.pt',weights_only=True));dec=RigDecoders(channels=8).to('mps').eval();checkpoint=root/f'results/round2_vae/rig/seed{seed}/best.pt';dec.load_state_dict(torch.load(checkpoint,weights_only=True));stats=dict(np.load(folder/'latent_stats.npz'));rows=[]
    with torch.no_grad():
        for i,r in enumerate(records):
            name=r['filename'];base=root/'data/external/dino_cohort'/Path(name).stem;target=targets[name];image=torch.tensor(np.load(base/'features.npz')['image_features'][0].mean(0)[None],device='mps');torch.manual_seed(seed*10000+i)
            pred,diag=generate_asset(image,flows,field,dec,stats,cfg=args.cfg)
            dest=root/f'data/external/generated/seed{seed}' if args.output_name=='generation' else root/f'data/external/{args.output_name}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            row=dict(filename=name,generation=diag)
            if pred is not None:
                np.savez_compressed(dest/name,**pred);row['generation'].update(mesh_chamfer=mesh_metric(pred['vertices'],pred['faces'],target['points']),skeleton=rignet_metrics(pred['joints'],pred['parents'],target['joints'],target['parents']),skin_rows_sum_max_error=float(np.abs(pred['weights'].sum(-1)-1).max()) if pred['weights'].shape[1] and len(pred['weights']) else None)
            # This explicitly separate diagnostic feeds ground-truth active positions/rig-derived encoded features.
            latent={}
            for mode in ['surface','rig']:
                d=np.load(root/f'data/external/vae4000_latents/seed{seed}'/(mode+'_'+name));ix=fps_indices(d['coords'],48);latent[mode]=(torch.tensor(d['features'][ix][None],device='mps'),torch.tensor(((d['coords'][ix]+.5)/64-.5).astype('float32')[None],device='mps'))
            sf,sc=latent['surface'];rf,rc=latent['rig'];q,c=regular_grid(16,'mps');sdf=torch.cat([field(x[None],sf,sc,rf,rc)[0] for x in q.split(512)]);v,f,_,_=FlexiCubes('cpu')(q.cpu(),sdf.cpu(),c.cpu(),16);actual=np.load(base/'sdf.npz')['sdf']
            row['gt_latent_reconstruction']=dict(input_scope='GT sparse positions and GT rig/skin-derived encoded DINO latents; not generation',sdf_mae=float(np.abs(sdf.cpu().numpy()-actual).mean()),mesh_chamfer=mesh_metric(v.numpy(),f.numpy(),target['points']),vertices=len(v),faces=len(f));np.savez_compressed(dest/('reconstruction_'+name),vertices=v.numpy(),faces=f.numpy());rows.append(row);print(seed,name,json.dumps(row),flush=True)
    report=dict(seed=seed,steps=25,cfg=args.cfg,view=0,scope='Strict image-only inference; target geometry accessed only outside generate_asset for evaluation. GT-latent reconstruction reported separately.',decoder_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),assets=rows)
    output=root/f'results/round2/{args.output_name}/seed{seed}';output.mkdir(parents=True,exist_ok=True)
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
