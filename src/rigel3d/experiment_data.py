"""Shared deterministic context loader for training and independent replay."""
import json
from pathlib import Path
import numpy as np
from .data import load_assets,fps_indices

def load_context_assets(root,config):
    base=load_assets(root);context=config.get('context','geometry')
    if not config.get('cohort',False) and context=='geometry':return base
    allowed={r['filename'] for r in json.loads((root/'results/round2/dino_features.json').read_text())['assets']};groups={s:[] for s in base}
    for split,rows in base.items():
        for original in rows:
            name=original['record']['filename']
            if name not in allowed:continue
            row=dict(original)
            if context!='geometry':
                for mode in ['surface','rig']:
                    if context=='trellis':
                        d=np.load(root/'data/external/dino_cohort'/Path(name).stem/'trellis_latents.npz');c=d[mode+'_coords'];f=d[mode+'_mu']
                    elif context=='vae':
                        d=np.load(root/f"data/external/{config.get('vae_run','vae4000')}_latents/seed{config['seed']}"/(mode+'_'+name));c=d['coords'];f=d['features']
                    else:
                        d=np.load(root/'data/external/dino_cohort'/Path(name).stem/'features.npz');c=d[mode+'_coords'];f=d[mode+'_features']
                    ix=fps_indices(c,48);row[mode+'_coords']=((c[ix]+.5)/64-.5).astype('float32');row[mode+'_features']=f[ix]
            groups[split].append(row)
    return groups
