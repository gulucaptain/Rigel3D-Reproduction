"""Validation-only EOS probability calibration, kept separate from threshold.5 main."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from rigel3d.data import load_assets,fps_indices
from rigel3d.decoders import RigDecoders
from rigel3d.trellis_runtime import generate_skeleton
from rigel3d.representation import decode_tokens
from rigel3d.metrics import rignet_metrics
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--joint-run',default='joint_frozen_surface');a=ap.parse_args();torch.set_num_threads(4);allowed={r['filename'] for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets']};groups={s:[r for r in rows if r['record']['filename'] in allowed and len(r['tokens'])] for s,rows in load_assets(root).items()}
for seed in [0,1,2]:
    folder=root/f'results/round3/{a.joint_run}/seed{seed}';assert (folder/'training.json').exists();state=torch.load(folder/'best.pt',weights_only=True);dec=RigDecoders(channels=8).eval();dec.load_state_dict({k.removeprefix('rig_decoders.'):v for k,v in state.items() if k.startswith('rig_decoders.')});heads={}
    for m in ['surface','rig']:
        h=torch.nn.Linear(768,16);h.load_state_dict({k:state[m+'_head.'+k] for k in ['weight','bias']});heads[m]=h.eval()
    def context(row):
        base=root/'data/external/dino_cohort'/Path(row['record']['filename']).stem;d=np.load(base/'trellis_latents.npz');lat={}
        for m in ['surface','rig']:
            ix=fps_indices(d[m+'_coords'],48);mu=heads[m](torch.tensor(d[m+'_hidden'][ix].astype('float32')))[:,:8];c=torch.tensor(((d[m+'_coords'][ix]+.5)/64-.5).astype('float32'));lat[m]=(mu[None],c[None])
        return dec.encoder(*lat['surface'],*lat['rig'])
    with torch.no_grad():
        val=[]
        for row in groups['val']:
            s,r=context(row);n=len(row['tokens']);out=dec.skeleton(torch.tensor(row['tokens'])[None],s,r);p=out['stop'][0].softmax(-1)[:,1].numpy();val.append(dict(filename=row['record']['filename'],continue_probabilities=p[:n].tolist(),end_probability=float(p[n])))
        grid=[]
        for threshold in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
            tpr=np.mean([v['end_probability']>=threshold for v in val]);tnr=np.mean([np.mean(np.array(v['continue_probabilities'])<threshold) for v in val]);grid.append(dict(threshold=threshold,balanced_accuracy=float((tpr+tnr)/2),end_recall=float(tpr),continue_accuracy=float(tnr)))
        selected=max(grid,key=lambda v:(v['balanced_accuracy'],-abs(v['threshold']-.5)))['threshold'];tests=[]
        for row in groups['test']:
            s,r=context(row);item=dict(filename=row['record']['filename'],gt_count=len(row['tokens']))
            for label,threshold in [('main',.5),('validation_calibrated',selected)]:
                tokens,diag=generate_skeleton(dec.skeleton,s,r,threshold);j,p=decode_tokens(tokens.numpy());item[label]=dict(**diag,count=len(j),count_abs_error=abs(len(j)-len(row['tokens'])),skeleton=rignet_metrics(j,p,row['joints'],row['parents']))
            tests.append(item)
    (folder/'eos_audit.json').write_text(json.dumps(dict(seed=seed,scope='GT-context reconstruction diagnostic. Threshold selected only from6 validation assets using teacher-forced balanced EOS/continue accuracy; fixed.5 stays main. No GT count/prefix at free inference.',grid=grid,selected_threshold=selected,validation=val,test=tests),indent=2)+'\n');print(seed,selected,flush=True)
