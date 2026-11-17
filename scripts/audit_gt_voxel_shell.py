"""GT-surface-support shell control: no DINO, no learned decoder, not generation."""
import json,itertools,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.trellis_native import Sparse
from rigel3d.trellis_mesh import extract_mesh
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);records=[r for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets'] if r['split']=='test'];rows=[]
for r in records:
    base=root/'data/external/dino_cohort'/Path(r['filename']).stem;d=np.load(base/'trellis_latents.npz');c=d['surface_coords'];offset=np.array(list(itertools.product(range(2),repeat=3)))
    for _ in range(2):c=(c[:,None]*2+offset[None]).reshape(-1,3)
    start=time.perf_counter()
    with torch.no_grad():v,f,diag=extract_mesh(Sparse(torch.zeros(len(c),101),c))
    cd=None
    if len(f):
        v=v.numpy();f=f.numpy();tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1);rng=np.random.default_rng(777);ids=rng.choice(len(tri),2048,p=area/area.sum());uv=rng.random((2048,2));uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1];p=tri[ids,0]+uv[:,:1]*(tri[ids,1]-tri[ids,0])+uv[:,1:]*(tri[ids,2]-tri[ids,0]);target=np.load(root/'data/external/prepared'/r['filename'])['points'];cd=float((cKDTree(p).query(target)[0].mean()+cKDTree(target).query(p)[0].mean())/2)
    rows.append(dict(filename=r['filename'],vertices=len(v),faces=len(f),mesh_chamfer=cd,elapsed_s=time.perf_counter()-start));print(json.dumps(rows[-1]),flush=True)
(root/'results/round3/gt_voxel_shell_control.json').write_text(json.dumps(dict(scope='GT64^3 surface occupied cells, exact256 subdivision, all101 predicted channels set to zero: default bias makes a shell near supplied GT positions. No DINO/learned decoder. Not generation; probes how much CD is explained by GT support. May contain double surfaces and does not prove valid geometry.',assets=rows),indent=2)+'\n')
