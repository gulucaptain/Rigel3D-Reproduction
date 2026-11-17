"""Strict single-image evaluation; targets read only AFTER generator returns."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.data import load_assets
from rigel3d.joint_trellis import JointTrellis,FullJointTrellis
from rigel3d.trellis_flow import CoupledFlows
from rigel3d.trellis_runtime import generate_image_only,load_occupancy_decoder
from rigel3d.metrics import rignet_metrics
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);ap.add_argument('--cfg',type=float,default=5);ap.add_argument('--name',default='generation');ap.add_argument('--limit',type=int,default=100);ap.add_argument('--full-joint',action='store_true');ap.add_argument('--joint-run',default='joint_frozen_surface');ap.add_argument('--flow-run',default='flows_frozen_surface');a=ap.parse_args();torch.set_num_threads(4);records=[r for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets'] if r['split']=='test'][:a.limit];targets={r['record']['filename']:r for r in load_assets(root)['test']}
def mesh_metric(v,f,target):
    if not len(f):return None
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)
    if area.sum()<1e-12:return None
    rng=np.random.default_rng(777);ids=rng.choice(len(f),2048,p=area/area.sum());uv=rng.random((2048,2));uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1];p=tri[ids,0]+uv[:,:1]*(tri[ids,1]-tri[ids,0])+uv[:,1:]*(tri[ids,2]-tri[ids,0]);return float((cKDTree(p).query(target)[0].mean()+cKDTree(target).query(p)[0].mean())/2)
for seed in a.seeds:
    folder=root/f'results/round3/{a.flow_run}/seed{seed}';assert (folder/'training.json').exists();flows=CoupledFlows().to('mps').eval();flows.load_state_dict(torch.load(folder/'best.pt',weights_only=True));joint=(FullJointTrellis if a.full_joint else JointTrellis)(root/'data/external/trellis/weights').eval()
    if a.full_joint:
        manifest=json.load(open(root/f'results/round3/{a.joint_run}/seed{seed}/model_manifest.json'));joint.load_state_dict(torch.load(root/manifest['checkpoint'],weights_only=True),strict=True)
    else:joint.load_trainable_state(torch.load(root/f'results/round3/{a.joint_run}/seed{seed}/best.pt',weights_only=True))
    decoder=load_occupancy_decoder(root/'data/external/trellis/weights');stats=dict(np.load(folder/'stats.npz'));rows=[]
    for i,r in enumerate(records):
        name=r['filename'];base=root/'data/external/dino_cohort'/Path(name).stem;image=torch.tensor(np.load(base/'trellis_image_tokens.npz')['tokens'][0].astype('float32'),device='mps')[None];torch.manual_seed(seed*10000+i);start=time.perf_counter();pred,diag=generate_image_only(image,flows,joint,decoder,stats,cfg=a.cfg)
        # All target access is confined to scoring outside generate_image_only().
        target=targets[name];gt=np.load(base/'features.npz');metrics={}
        for mode in ['surface','rig']:
            mask=pred[mode+'_occupancy'];coords=gt[mode+'_coords'];truth=np.zeros((64,64,64),bool);truth[coords[:,0],coords[:,1],coords[:,2]]=True;inter=(mask&truth).sum();union=(mask|truth).sum();metrics[mode+'_occupancy']=dict(iou=float(inter/max(union,1)),precision=float(inter/max(mask.sum(),1)),recall=float(inter/max(truth.sum(),1)),predicted_count=int(mask.sum()),target_count=int(truth.sum()))
        target_occ=np.load(base/'occupancy_pretrained.npz')
        for branch,mode in enumerate(['surface','rig']):
            normalized=(target_occ[mode+'_mu']-stats[mode+'_occ_mean'][:,None,None,None])/stats[mode+'_occ_std'][:,None,None,None];metrics[mode+'_occupancy_latent_mse']=float(np.square(pred['occupancy_latents'][0,branch*8:(branch+1)*8]-normalized).mean())
        if 'vertices' in pred:
            metrics['mesh_chamfer']=mesh_metric(pred['vertices'],pred['faces'],target['points']);metrics['skeleton']=rignet_metrics(pred['joints'],pred['parents'],target['joints'],target['parents'])
        else:metrics['mesh_chamfer']=None
        dest=root/f'data/external/trellis/{a.name}/seed{seed}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/name,**pred);rec=dict(filename=name,**diag,**metrics,elapsed_s=time.perf_counter()-start);rows.append(rec);print(seed,json.dumps(rec),flush=True)
    dest=root/f'results/round3/{a.name}/seed{seed}';dest.mkdir(parents=True,exist_ok=False);(dest/'evaluation.json').write_text(json.dumps(dict(seed=seed,cfg=a.cfg,steps=25,view=0,scope='Single input image only. Occupancy64^3 and local coordinates/latents predicted; actual pretrained256 mesh + joint adapted rig. Every empty/budget failure retained. Target geometry/rig read only outside generation function for scoring.',assets=rows),indent=2,allow_nan=False)+'\n');del flows,joint,decoder;torch.mps.empty_cache()
