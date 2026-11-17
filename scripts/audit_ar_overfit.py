"""Single TRAIN-asset memorization audit, not held-out generation performance."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from rigel3d.data import load_assets
from rigel3d.decoders import RigDecoders,skeleton_loss
from rigel3d.metrics import rignet_metrics
from rigel3d.representation import decode_tokens
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2000);a=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(0);row=next(r for r in load_assets(root)['train'] if len(r['tokens']))
device='mps'
if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
def t(k):return torch.tensor(row[k],device=device)[None]
model=RigDecoders().to(device);opt=torch.optim.AdamW(list(model.encoder.parameters())+list(model.skeleton.parameters()),lr=1e-4)
target=t('tokens');n=len(row['tokens']);trace=[];start=time.perf_counter()
for step in range(a.steps):
    model.train();opt.zero_grad(set_to_none=True)
    s,r=model.encoder(t('surface_features'),t('surface_coords'),t('rig_features'),t('rig_coords'))
    out=model.skeleton(target,s,r);loss,terms=skeleton_loss(out,target,torch.tensor([n],device=device))
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();opt.step()
    if step%200==0 or step==a.steps-1:
        model.eval()
        with torch.no_grad():
            s,r=model.encoder(t('surface_features'),t('surface_coords'),t('rig_features'),t('rig_coords'))
            out=model.skeleton(target,s,r);tokens,diag=model.skeleton.generate(s,r);j,p=decode_tokens(tokens.cpu().numpy())
            rec=dict(step=step,loss=float(loss.detach().cpu()),coordinate_accuracy=float((out['coords'][0,:n].argmax(-1)==target[0,:,:3]).float().mean().cpu()),
                parent_accuracy=float((out['parent'][0,:n].argmax(-1)==target[0,:,3]).float().mean().cpu()),
                eos_at_true_end=bool(out['stop'][0,n].argmax()==1),generated_count=len(j),gt_count=n,
                free_metrics=rignet_metrics(j,p,row['joints'],row['parents']),**diag)
        trace.append(rec);print(json.dumps(rec),flush=True)
folder=root/'results/round2';folder.mkdir(exist_ok=True)
(folder/'single_asset_overfit.json').write_text(json.dumps(dict(scope='Single TRAIN asset memorization, GT rig reconstruction context; no held-out claim',asset_name=row['record']['asset_name'],seed=0,steps=a.steps,elapsed_s=time.perf_counter()-start,trace=trace),indent=2)+'\n')
