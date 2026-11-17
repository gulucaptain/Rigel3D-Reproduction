"""Coordinate support controls; no new model/threshold selected from test results."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.data import load_assets,fps_indices
from rigel3d.decoders import RigDecoders
from rigel3d.trellis_runtime import generate_skeleton
from rigel3d.representation import decode_tokens
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--joint-run',default='joint_frozen_surface');a=ap.parse_args();torch.set_num_threads(4);allowed={r['filename'] for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets']};tests=[r for r in load_assets(root)['test'] if r['record']['filename'] in allowed]
for seed in [0,1,2]:
    folder=root/f'results/round3/{a.joint_run}/seed{seed}';assert (folder/'training.json').exists();state=torch.load(folder/'best.pt',weights_only=True);dec=RigDecoders(channels=8).eval();dec.load_state_dict({k.removeprefix('rig_decoders.'):v for k,v in state.items() if k.startswith('rig_decoders.')});heads={}
    for mode in ['surface','rig']:
        h=torch.nn.Linear(768,16);h.load_state_dict({k:state[mode+'_head.'+k] for k in ['weight','bias']});heads[mode]=h.eval()
    rows=[]
    with torch.no_grad():
        for row in tests:
            base=root/'data/external/dino_cohort'/Path(row['record']['filename']).stem;d=np.load(base/'trellis_latents.npz');lat={}
            for mode in ['surface','rig']:
                ix=fps_indices(d[mode+'_coords'],48);mu=heads[mode](torch.tensor(d[mode+'_hidden'][ix].astype('float32')))[:,:8];c=torch.tensor(((d[mode+'_coords'][ix]+.5)/64-.5).astype('float32'));lat[mode]=(mu[None],c[None])
            s,r=dec.encoder(*lat['surface'],*lat['rig']);target=torch.tensor(row['tokens'])[None];n=len(row['tokens']);out=dec.skeleton(target,s,r);logp=out['coords'][0,:n].log_softmax(-1);independent=logp.argmax(-1).numpy()/127-.5
            candidate=((d['rig_coords']+.5)/64-.5).astype('float32');q=torch.tensor(np.rint((candidate+.5)*127).astype(int));score=sum(logp[:,axis,q[:,axis]] for axis in range(3));supported=candidate[score.argmax(-1).numpy()];truth=row['tokens'][:,:3]/127-.5;tree=cKDTree(candidate)
            tokens,diag=generate_skeleton(dec.skeleton,s,r);free,_=decode_tokens(tokens.numpy());distance=tree.query(free)[0];rows.append(dict(filename=row['record']['filename'],teacher_matched_l2=float(np.linalg.norm(independent-truth,axis=-1).mean()),gt_rig_supported_teacher_l2=float(np.linalg.norm(supported-truth,axis=-1).mean()),gt_joint_to_rig_voxel_distance=float(tree.query(truth)[0].mean()),free_joint_to_rig_voxel_distance=float(distance.mean()),free_fraction_within_one_voxel=float((distance<=1/64).mean()),capped=diag['capped']))
    (folder/'coordinate_audit.json').write_text(json.dumps(dict(seed=seed,scope='GT-context diagnostic. Candidate coordinates from supplied GT-rig voxels are an oracle support control, not the paper decoder and never main generation. Teacher forcing uses GT prefix; free statistics use BOS only.',assets=rows),indent=2)+'\n');print(seed,flush=True)
