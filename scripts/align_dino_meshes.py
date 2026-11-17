"""Geometry-only similarity/ICP alignment; reject mismatched source poses explicitly."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
root=Path(__file__).resolve().parents[1];manifest=json.loads((root/'results/round2/mesh_export.json').read_text());records=[]
for rec in manifest['assets']:
    row=dict(rec)
    if rec.get('mesh_status')!='exported':records.append(row);continue
    folder=root/'data/external/dino_cohort'/Path(rec['filename']).stem;mesh=dict(np.load(folder/'raw_mesh.npz'));target=np.load(root/'data/external/prepared'/rec['filename'])['points'].astype(float)
    v=mesh['vertices'].astype(float);f=mesh['faces'];rng=np.random.default_rng(312)
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1);valid=area>1e-15
    if not valid.any():row['alignment_status']='degenerate';records.append(row);continue
    tri=tri[valid];area=area[valid];ids=rng.choice(len(tri),8192,p=area/area.sum());uv=rng.random((8192,2));uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1]
    samples=tri[ids,0]+uv[:,:1]*(tri[ids,1]-tri[ids,0])+uv[:,1:]*(tri[ids,2]-tri[ids,0])
    scale=np.ptp(target,axis=0).max()/np.ptp(samples,axis=0).max();translation=(target.min(0)+target.max(0))/2-scale*(samples.min(0)+samples.max(0))/2
    r=np.eye(3);tree=cKDTree(target)
    for _ in range(40):
        moved=scale*samples@r.T+translation;dist,idx=tree.query(moved);keep=dist<=np.quantile(dist,.85)
        a=moved[keep];b=target[idx[keep]];ac=a.mean(0);bc=b.mean(0);u,_,vt=np.linalg.svd((a-ac).T@(b-bc));fix=np.eye(3);fix[-1,-1]=np.linalg.det(vt.T@u.T)
        rr=vt.T@fix@u.T;r=rr@r;translation=rr@translation+bc-rr@ac
    aligned=scale*v@r.T+translation;moved=scale*samples@r.T+translation
    forward=tree.query(moved)[0];backward=cKDTree(moved).query(target)[0];median=float(np.median(np.r_[forward,backward]));p90=float(np.quantile(np.r_[forward,backward],.9))
    # Fixed pre-training registration gate; never tuned by decoder/test scores.
    good=median<.012 and p90<.035 and np.abs(aligned).max()<.65
    row.update(alignment_status='accepted' if good else 'rejected_pose_or_geometry',alignment_median=median,alignment_p90=p90,
        alignment_scope='Similarity/rigid ICP using surface points only; no joints or skin weights used; fixed gate median<.012, p90<.035, bounds<.65')
    if good:np.savez_compressed(folder/'aligned_mesh.npz',vertices=aligned.astype(np.float32),faces=f)
    records.append(row)
(root/'results/round2/mesh_alignment.json').write_text(json.dumps(dict(assets=records),indent=2)+'\n')
print(json.dumps({s:sum(r['split']==s and r.get('alignment_status')=='accepted' for r in records) for s in ['train','val','test']}))
