"""Measure generated latent shift and predicted occupancy, never choose a model."""
import json
from pathlib import Path
import numpy as np,torch
from rigel3d.generation import FourFlows
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);records=[r for r in json.load(open(root/'results/round2/dino_features.json'))['assets'] if r['split']=='test'];rows=[]
with torch.no_grad():
    for seed in [0,1,2]:
        f=FourFlows().to('mps').eval();f.load_state_dict(torch.load(root/f'results/round2/generation/seed{seed}/flows.pt',weights_only=True))
        for i,r in enumerate(records):
            d=np.load(root/'data/external/dino_cohort'/Path(r['filename']).stem/'features.npz');image=torch.tensor(d['image_features'][0].mean(0)[None],device='mps');torch.manual_seed(seed*10000+i);pred,diag=f.sample(image);row=dict(seed=seed,filename=r['filename'],status=diag)
            if pred:
                for mode in ['surface','rig']:
                    p=pred[mode];rms=float(p['features'].square().mean().sqrt().cpu());coarse=np.clip(d[mode+'_coords']//4,0,15);target={tuple(c) for c in coarse};actual={tuple(c) for c in np.round((p['coords'].cpu().numpy()+.5)*16-.5).astype(int)}
                    row[mode]=dict(normalized_latent_rms=rms,coarse_occupancy_iou=len(target&actual)/len(target|actual),target_coarse_cells=len(target),predicted_coarse_cells=len(actual))
            rows.append(row)
report=dict(scope='CFG5 held-out diagnostic only; normalized training latent RMS is 1 by construction. Target coordinates are accessed solely after image-only sample() for scoring. No checkpoint/config selection.',assets=rows)
(root/'results/round2/generation_shift_audit.json').write_text(json.dumps(report,indent=2)+'\n')
for mode in ['surface','rig']:print(mode,{k:float(np.mean([r[mode][k] for r in rows if mode in r])) for k in ['normalized_latent_rms','coarse_occupancy_iou']})
