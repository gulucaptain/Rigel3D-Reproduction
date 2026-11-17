"""Full256 GT-conditioned joint reconstruction, free skeleton and oracle-bone skins."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from rigel3d.data import load_assets,fps_indices
from rigel3d.joint_trellis import JointTrellis,FullJointTrellis
from rigel3d.trellis_runtime import decode_rigged_asset,generate_skeleton
from rigel3d.metrics import rignet_metrics,skinning_metrics
from rigel3d.representation import decode_tokens
from scipy.spatial import cKDTree
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);ap.add_argument('--full-joint',action='store_true');ap.add_argument('--joint-run',default='joint_frozen_surface');a=ap.parse_args();torch.set_num_threads(4);allowed={r['filename'] for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets']};groups={s:[r for r in rows if r['record']['filename'] in allowed] for s,rows in load_assets(root).items()}
def mesh_chamfer(v,f,target):
    if not len(f):return None
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1);rng=np.random.default_rng(777);ids=rng.choice(len(f),2048,p=area/area.sum());uv=rng.random((2048,2));uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1];p=tri[ids,0]+uv[:,:1]*(tri[ids,1]-tri[ids,0])+uv[:,1:]*(tri[ids,2]-tri[ids,0]);return float((cKDTree(p).query(target)[0].mean()+cKDTree(target).query(p)[0].mean())/2)
for seed in a.seeds:
    folder=root/f'results/round3/{a.joint_run}/seed{seed}';assert (folder/'training.json').exists();model=(FullJointTrellis if a.full_joint else JointTrellis)(root/'data/external/trellis/weights').eval()
    if a.full_joint:
        manifest=json.load(open(folder/'model_manifest.json'));model.load_state_dict(torch.load(root/manifest['checkpoint'],weights_only=True),strict=True)
    else:model.load_trainable_state(torch.load(folder/'best.pt',weights_only=True))
    rows=[]
    with torch.no_grad():
        for row in groups['test']:
            name=row['record']['filename'];base=root/'data/external/dino_cohort'/Path(name).stem;d=np.load(base/'trellis_latents.npz');post={};start=time.perf_counter()
            if a.full_joint:
                post['surface']=torch.tensor(d['surface_mu'],device='mps');rf=np.load(base/'trellis_features.npz')['rig_features'].astype('float32');post['rig']=model.encode_rig(torch.tensor(rf,device='mps'),d['rig_coords'],False)[0].feats
            else:
                for mode in ['surface','rig']:post[mode]=model.posterior(torch.tensor(d[mode+'_hidden'].astype('float32'),device='mps'),mode,False)[0]
            pred,diag=decode_rigged_asset(model,post['surface'],d['surface_coords'],post['rig'],d['rig_coords'],skin=False);rec=dict(filename=name,input_scope='GT occupied positions and GT-rig/skin-derived DINO features; free AR prefix and count. Reconstruction only.',mesh_chamfer=mesh_chamfer(pred['vertices'],pred['faces'],row['points']),**diag)
            if len(row['tokens']):rec['skeleton']=rignet_metrics(pred['joints'],pred['parents'],row['joints'],row['parents']);rec['count_abs_error']=abs(len(pred['joints'])-len(row['joints']))
            si=fps_indices(d['surface_coords'],48);ri=fps_indices(d['rig_coords'],48);sc=torch.tensor(((d['surface_coords'][si]+.5)/64-.5).astype('float32'),device='mps')[None];rc=torch.tensor(((d['rig_coords'][ri]+.5)/64-.5).astype('float32'),device='mps')[None];s,r=model.rig_decoders.encoder(post['surface'][si][None],sc,post['rig'][ri][None],rc)
            target=torch.tensor(row['tokens'],device='mps')[None];out=model.rig_decoders.skeleton(target,s,r);n=len(row['tokens'])
            rec['teacher_forced']=dict(coordinate_accuracy=float((out['coords'][0,:n].argmax(-1)==target[0,:,:3]).float().mean().cpu()),eos_true_end_probability=float(out['stop'][0,n].softmax(-1)[1].cpu()),continue_accuracy=float((out['stop'][0,:n].argmax(-1)==0).float().mean().cpu()))
            rr=np.random.default_rng(37);ix=rr.choice(np.flatnonzero(row['skin_valid']),512,replace=False)
            def t(v):return torch.tensor(v,device='mps')[None]
            w=model.rig_decoders.skinning(t(row['points'][ix]),t(row['normals'][ix]),t(row['bones'][:,0]),t(row['bones'][:,1]),s,r)[0].cpu().numpy();rec['gt_bone_skin']=skinning_metrics(w,row['weights'][ix]);rec['elapsed_s']=time.perf_counter()-start;rows.append(rec)
            dest=root/f'data/external/trellis/{a.joint_run}_reconstruction/seed{seed}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/name,**pred);print(seed,json.dumps(rec),flush=True)
    (folder/'evaluation.json').write_text(json.dumps(dict(seed=seed,scope='All test cases; GT-conditioned reconstruction only. Mesh uses full surface voxel set at256 extraction. GT-bone skin accuracy separate from generated skeleton.',assets=rows),indent=2,allow_nan=False)+'\n');del model;torch.mps.empty_cache()
