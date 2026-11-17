"""GT-conditioned official pretrained encoder+mesh decoder reconstruction only."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from rigel3d.trellis_native import Sparse,load_pretrained
from rigel3d.trellis_mesh import extract_mesh
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--split',default='test');ap.add_argument('--limit',type=int,default=100);a=ap.parse_args();torch.set_num_threads(4);model=load_pretrained('mesh',root/'data/external/trellis/weights','mps').float().eval();records=[r for r in json.load(open(root/'results/round3/dino_features.json'))['assets'] if r['split']==a.split][:a.limit];rows=[]
for r in records:
    base=root/'data/external/dino_cohort'/Path(r['filename']).stem;d=np.load(base/'trellis_latents.npz');start=time.perf_counter()
    with torch.no_grad():
        out=model(Sparse(torch.tensor(d['surface_mu'],device='mps'),d['surface_coords']));sdf=out.feats[:,:8]-1/256;row=dict(filename=r['filename'],split=r['split'],decoder_cubes=len(out.coords),sdf_min=float(sdf.min().cpu()),sdf_max=float(sdf.max().cpu()),negative_corners=int((sdf<0).sum().cpu()),network_s=time.perf_counter()-start)
        v,f,diag=extract_mesh(out)
    dest=root/'data/external/trellis/reconstruction';dest.mkdir(exist_ok=True);np.savez_compressed(dest/r['filename'],vertices=v.numpy(),faces=f.numpy());target=np.load(root/'data/external/prepared'/r['filename'])['points'];cd=None
    if len(f):
        tri=v.numpy()[f.numpy()];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1);rng=np.random.default_rng(777);ids=rng.choice(len(tri),2048,p=area/area.sum());uv=rng.random((2048,2));uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1];p=tri[ids,0]+uv[:,:1]*(tri[ids,1]-tri[ids,0])+uv[:,1:]*(tri[ids,2]-tri[ids,0]);cd=float((cKDTree(p).query(target)[0].mean()+cKDTree(target).query(p)[0].mean())/2)
    row.update(vertices=len(v),faces=len(f),mesh_chamfer=cd,elapsed_s=time.perf_counter()-start,extraction=diag);rows.append(row);print(json.dumps(row),flush=True)
(root/f'results/round3/pretrained_reconstruction_{a.split}.json').write_text(json.dumps(dict(scope='GT occupied surface coordinates and encoded original multiview DINO-L features; official pretrained weights, native FP32 kernels, full256 mesh extraction. This is reconstruction, not image generation.',assets=rows),indent=2)+'\n')
