"""Audit all held-out predictions with pinned RigNet-compatible metrics.

Preserve v1 continuous-segment diagnostics; no checkpoint/test-based selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rigel3d.data import load_assets
from rigel3d.experiment_data import load_context_assets
from rigel3d.decoders import RigDecoders
from rigel3d.metrics import rignet_metrics,skinning_metrics
from rigel3d.representation import decode_tokens
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--suites',nargs='+',default=['round1','round1_balanced']);args=ap.parse_args()
torch.set_num_threads(4);groups=load_assets(root)
for suite in args.suites:
    for path in sorted((root/'results'/suite).glob('*/seed*/evaluation.json')):
        rec=json.loads(path.read_text());config=rec['config'];groups=load_context_assets(root,config);model=RigDecoders(config['width'],config['layers'],96,channels={'geometry':16,'dino':384,'vae':8,'trellis':8}[config.get('context','geometry')]).eval()
        model.load_state_dict(torch.load(path.parent/'best.pt',weights_only=True,map_location='cpu'))
        audited={}
        with torch.no_grad():
            for split,rows in [('train_monitor',groups['train'][:16]),('val',groups['val']),('test',groups['test'])]:
                records=[]
                for i,(row,legacy) in enumerate(zip(rows,rec['splits'][split])):
                    assert legacy['asset_name']==row['record']['asset_name']
                    predfile=root/'data/external/predictions'/suite/config['mode']/f"seed{config['seed']}"/split/f'{i:03}.npz'
                    pred=dict(np.load(predfile));j,p=decode_tokens(pred['tokens'])
                    r=dict(asset_name=legacy['asset_name'],skin=skinning_metrics(pred['weights'],row['weights'][pred['point_ids']]),
                        generated_joint_count=len(j),capped=bool(len(j)==96),skeleton_eligible=bool(len(row['tokens'])))
                    if len(row['tokens']):
                        r['skeleton']=rignet_metrics(j,p,row['joints'],row['parents'])
                        # Audit 4x denser sampling separately; not choose favorable metric.
                        dense=rignet_metrics(j,p,row['joints'],row['parents'],step=.00125)
                        r['metric_sensitivity']=dict(j2b_dense=dense['j2b'],b2b_dense=dense['b2b'],
                            j2b_abs_difference=abs(dense['j2b']-r['skeleton']['j2b']),b2b_abs_difference=abs(dense['b2b']-r['skeleton']['b2b']))
                        def t(k):return torch.tensor(row[k])[None]
                        surf,rig=model.encoder(t('surface_features'),t('surface_coords'),t('rig_features' if config['mode']=='rig' else 'geometry_rig_features'),t('rig_coords'))
                        target=t('tokens');out=model.skeleton(target,surf,rig);n=target.shape[1]
                        r['teacher_forced']=dict(coordinate_accuracy=float((out['coords'][0,:n].argmax(-1)==target[0,:,:3]).float().mean()),
                            parent_accuracy=float((out['parent'][0,:n].argmax(-1)==target[0,:,3]).float().mean()),
                            eos_at_true_end=bool(out['stop'][0,n].argmax()==1),
                            continue_accuracy=float((out['stop'][0,:n].argmax(-1)==0).float().mean()))
                        tj=out['coords'][0,:n].argmax(-1).numpy()/127-.5;tp=out['parent'][0,:n].argmax(-1).numpy()
                        r['teacher_forced_geometry_diagnostic']=rignet_metrics(tj,tp,row['joints'],row['parents'])
                        qj,qp=decode_tokens(row['tokens'])
                        r['quantized_gt_oracle']=rignet_metrics(qj,qp,row['joints'],row['parents'])
                        if len(j)>=n:
                            r['gt_count_prefix_diagnostic']=rignet_metrics(j[:n],p[:n],row['joints'],row['parents'])
                    records.append(r)
                audited[split]=records
        report=dict(scope='Held-out reconstruction proxy, not image-conditioned generation; skinning uses supplied GT bones and same-index weights only within that oracle-bone task',
            metric_version=2,legacy='evaluation.json retains v1 analytic J2B and equal-bone quadrature B2B; not used as paper-equivalent metrics',
            skeleton_definition='Un-squared symmetric mean/2; RigNet bone sampling round(length/0.005), on the explicit unit geometry box',
            skin_definition='Element mean L1 and vector-sum L1 both retained; L2 vector norm separate from mean squared element; threshold=.05',
            config=config,best_step=rec['best_step'],splits=audited)
        (path.parent/'evaluation_metrics_v2.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(str(path.parent),flush=True)
