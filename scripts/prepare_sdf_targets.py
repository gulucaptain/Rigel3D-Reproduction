"""Blender BVH exact nearest-triangle signed-normal field (weak sign on open meshes)."""
import json
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
root=Path(__file__).resolve().parents[1];records=json.loads((root/'results/round2/dino_features.json').read_text())['assets'];rows=[]
a=np.linspace(-.5,.5,17);queries=np.stack(np.meshgrid(a,a,a,indexing='ij'),-1).reshape(-1,3)
for row in records:
    folder=root/'data/external/dino_cohort'/Path(row['filename']).stem;d=np.load(folder/'aligned_mesh.npz');v=d['vertices'];f=d['faces'];tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);sdf=[]
    for p in queries:
        loc,n,idx,dist=tree.find_nearest(Vector(p));sdf.append(dist*(1 if (Vector(p)-loc).dot(n)>=0 else -1))
    sdf=np.asarray(sdf,np.float32).clip(-.15,.15);np.savez_compressed(folder/'sdf.npz',points=queries.astype('float32'),sdf=sdf)
    rows.append(dict(filename=row['filename'],split=row['split'],negative_fraction=float((sdf<0).mean())))
(root/'results/round2/sdf_targets.json').write_text(json.dumps(dict(resolution=16,sign='nearest oriented triangle normal; not watertight parity; unreliable signs for open/inconsistent meshes',assets=rows),indent=2)+'\n')
