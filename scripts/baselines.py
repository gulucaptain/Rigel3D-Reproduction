"""Observation-only skin-distance and training-asset retrieval controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from rigel3d.data import load_assets,fps_indices
from rigel3d.metrics import skinning_metrics,point_segment_distance,rignet_metrics
root=Path(__file__).resolve().parents[1];g=load_assets(root)


def point_ids(row):
    rng=np.random.default_rng(int(hashlib.sha256(row['record']['asset_name'].encode()).hexdigest()[:8],16)+37)
    return rng.choice(np.flatnonzero(row['skin_valid']),512,replace=False)


def distance_skin(row,t):
    ids=point_ids(row);dist=point_segment_distance(row['points'][ids],row['bones'][:,0],row['bones'][:,1])
    pred=softmax(-dist**2/t**2,axis=-1)
    return skinning_metrics(pred,row['weights'][ids])


temperatures=[.01,.02,.05,.1,.2]
validation=[dict(temperature=t,l1_element=float(np.mean([distance_skin(r,t)['l1_element'] for r in g['val']]))) for t in temperatures]
best=min(validation,key=lambda r:r['l1_element'])['temperature']
train=[r for r in g['train'] if len(r['tokens'])]
representations=[r['points'][fps_indices(r['points'],64)] for r in train]
records=[]
for row in g['test']:
    pts=row['points'][fps_indices(row['points'],64)]
    scores=[]
    for ref in representations:
        d=cdist(pts,ref);scores.append(.5*(d.min(0).mean()+d.min(1).mean()))
    chosen=int(np.argmin(scores));r=train[chosen]
    records.append(dict(asset_name=row['record']['asset_name'],skin=distance_skin(row,best),
        retrieval_source=r['record']['asset_name'],skeleton=rignet_metrics(r['joints'],r['parents'],row['joints'],row['parents'])))
summary=dict(scope='Local controls, not published Rigel3D baselines. Skin uses GT bones only; skeleton retrieves a train rig by 64-point surface Chamfer, no test rig features used.',
    temperature_selection='Minimum mean elementwise L1 on validation only',validation=validation,selected_temperature=best,test=records)
(root/'results/round1/baselines.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(selected_temperature=best,skin_l1=float(np.mean([r['skin']['l1_element'] for r in records])),j2j=float(np.mean([r['skeleton']['j2j'] for r in records])))))
