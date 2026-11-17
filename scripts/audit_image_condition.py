"""Fixed-noise image ablation, not a semantic-fidelity score or selection signal."""
import json
from pathlib import Path
import numpy as np,torch
from rigel3d.generation import FourFlows
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);r=next(r for r in json.load(open(root/'results/round2/dino_features.json'))['assets'] if r['split']=='test');p=root/'data/external/dino_cohort'/Path(r['filename']).stem/'features.npz';image=torch.tensor(np.load(p)['image_features'][0].mean(0)[None],device='mps');checks=[]
with torch.no_grad():
    for seed in [0,1,2]:
        flows=FourFlows().to('mps').eval();flows.load_state_dict(torch.load(root/f'results/round2/generation/seed{seed}/flows.pt',weights_only=True));torch.manual_seed(99);a,da=flows.sample(image);torch.manual_seed(99);b,db=flows.sample(torch.zeros_like(image));row=dict(seed=seed,conditioned_status=da,zero_condition_status=db)
        if a and b:
            for mode in ['surface','rig']:
                ac={tuple(v) for v in a[mode]['coords'].cpu().numpy()};bc={tuple(v) for v in b[mode]['coords'].cpu().numpy()};row[mode+'_cell_jaccard']=len(ac&bc)/len(ac|bc)
        checks.append(row)
(root/'results/round2/conditioning_audit.json').write_text(json.dumps(dict(scope='One fixed test image; same noise with actual versus zero DINO condition. Demonstrates sensitivity only, not correct image alignment or generation quality.',checks=checks),indent=2)+'\n');print(json.dumps(checks))
