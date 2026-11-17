"""Independently replay validation losses and sampled skin predictions on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rigel3d.data import load_assets
from rigel3d.experiment_data import load_context_assets
from rigel3d.decoders import RigDecoders,skeleton_loss
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--suite',default='round1');args=ap.parse_args();torch.set_num_threads(4);groups=load_assets(root);checks=[]
for path in sorted((root/'results'/args.suite).glob('*/seed*/evaluation.json')):
    rec=json.loads(path.read_text());c=rec['config'];groups=load_context_assets(root,c);model=RigDecoders(c['width'],c['layers'],96,channels={'geometry':16,'dino':384,'vae':8,'trellis':8}[c.get('context','geometry')]).eval()
    assert hashlib.sha256((root/'results/anymate_preparation.json').read_bytes()).hexdigest()==c['source_manifest_sha256']
    model.load_state_dict(torch.load(path.parent/'best.pt',map_location='cpu',weights_only=True));vals=[];maximum=0.
    with torch.no_grad():
        for row,old in zip(groups['val'],rec['splits']['val']):
            rng=np.random.default_rng(int(hashlib.sha256(row['record']['asset_name'].encode()).hexdigest()[:8],16)+37)
            ix=rng.choice(np.flatnonzero(row['skin_valid']),c['points'],replace=False)
            def t(k):return torch.tensor(row[k])[None]
            rf=t('rig_features' if c['mode']=='rig' else 'geometry_rig_features')
            s,r=model.encoder(t('surface_features'),t('surface_coords'),rf,t('rig_coords'))
            w=model.skinning(t('points')[:,ix],t('normals')[:,ix],t('bones')[:,:,0],t('bones')[:,:,1],s,r)
            skin=-(t('weights')[:,ix]*w.clamp_min(1e-12).log()).sum(-1).mean()
            skel=torch.tensor(0.)
            if len(row['tokens']):
                tokens=t('tokens');out=model.skeleton(tokens,s,r);skel,_=skeleton_loss(out,tokens,torch.tensor([len(row['tokens'])]),c.get('balanced_stop',False))
            vals.append(float(skin+skel))
        got=float(np.mean(vals));delta=abs(got-rec['best_validation'])
        assert delta<2e-4,(str(path),got,rec['best_validation'])
        # Separate point count 512 exactly as in held-out evaluation.
        row=groups['test'][0];rng=np.random.default_rng(int(hashlib.sha256(row['record']['asset_name'].encode()).hexdigest()[:8],16)+37)
        ix=rng.choice(np.flatnonzero(row['skin_valid']),512,replace=False)
        def t(k):return torch.tensor(row[k])[None]
        s,r=model.encoder(t('surface_features'),t('surface_coords'),t('rig_features' if c['mode']=='rig' else 'geometry_rig_features'),t('rig_coords'))
        w=model.skinning(t('points')[:,ix],t('normals')[:,ix],t('bones')[:,:,0],t('bones')[:,:,1],s,r)[0].numpy()
        predictions=root/'data/external/predictions'/args.suite/c['mode']/f"seed{c['seed']}"/'test/000.npz'
        saved=dict(np.load(predictions));np.testing.assert_array_equal(ix,saved['point_ids']);maximum=float(np.abs(w-saved['weights']).max())
        assert maximum<2e-4,(str(path),maximum)
    checks.append(dict(source=str(path.relative_to(root)),cpu_validation=got,recorded_validation=rec['best_validation'],absolute_difference=delta,skin_prediction_max_difference=maximum,passed=True))
dest=root/'results'/args.suite/'replay_check.json';dest.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n');print(json.dumps(dict(passed=True,runs=len(checks))))
