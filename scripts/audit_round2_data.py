"""Integration/feature and GT-field resolution controls, never generation claims."""
import json
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.data import load_assets
from rigel3d.flexicubes import FlexiCubes,regular_grid
root=Path(__file__).resolve().parents[1];dino=json.loads((root/'results/round2/dino_features.json').read_text());names={r['filename'] for r in dino['assets']};groups=load_assets(root);coverage={s:dict(assets=sum(r['record']['filename'] in names for r in rs),skeleton_eligible=sum(r['record']['filename'] in names and len(r['tokens'])>0 for r in rs)) for s,rs in groups.items()};test={r['record']['filename']:r for r in groups['test']};oracle=[]
q,c=regular_grid(16)
for r in dino['assets']:
    if r['split']!='test':continue
    base=root/'data/external/dino_cohort'/Path(r['filename']).stem;sdf=np.load(base/'sdf.npz')['sdf'];v,f,_,_=FlexiCubes('cpu')(q,torch.tensor(sdf),c,16)
    # Vertex-based distances only in this diagnostic, explicitly named.
    chamfer=None
    if len(v):chamfer=float((cKDTree(v.numpy()).query(test[r['filename']]['points'])[0].mean()+cKDTree(test[r['filename']]['points']).query(v.numpy())[0].mean())/2)
    oracle.append(dict(filename=r['filename'],vertices=len(v),faces=len(f),vertex_chamfer_diagnostic=chamfer))
baseline={}
for seed in [0,1,2]:
    baseline[str(seed)]={}
    for m in ['surface','rig']:
        norm=np.load(root/f'results/round2/vae4000/seed{seed}/{m}/normalization.npz');vals=[]
        for r in dino['assets']:
            if r['split']=='test':
                x=np.load(root/'data/external/dino_cohort'/Path(r['filename']).stem/'features.npz')[m+'_features'];vals.append(float((((x-norm['mean'])/norm['std'])**2).mean()))
        baseline[str(seed)][m]=float(np.mean(vals))
report=dict(coverage=coverage,visibility=dict(min=float(min(r['observed_fraction'] for r in dino['assets'])),mean=float(np.mean([r['observed_fraction'] for r in dino['assets']]))),zero_latent_feature_baseline_mse=baseline,gt_sdf_extraction_oracle=oracle,scope='Integration controls only. GT oriented-normal SDF extraction oracle is not learned reconstruction or image generation.')
(root/'results/round2/data_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(coverage=coverage,visibility=report['visibility'],zero_baseline=baseline)))
