"""Initial representation/attention resource checks; no generative training."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from rigel3d.representation import bfs_tokens,decode_tokens,segment_voxels
from rigel3d.skinning import SkinningDecoder,soft_cross_entropy
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--mps',action='store_true');args=ap.parse_args()
torch.set_num_threads(4);rng=np.random.default_rng(2026);rows=[]
for res in [32,64,128,256]:
    errors=[]
    for _ in range(100):
        j=rng.uniform(-.5,.5,(32,3));p=np.array([0]+[rng.integers(i) for i in range(1,32)])
        tokens,order=bfs_tokens(j,p,res);decoded,_=decode_tokens(tokens,res)
        errors.extend(np.linalg.norm(decoded-j[order],axis=-1).tolist())
    rows.append(dict(resolution=res,joints=3200,mean_error=float(np.mean(errors)),max_error=float(np.max(errors)),bound=float(np.sqrt(3)/(2*(res-1)))))
torch.manual_seed(0);model=SkinningDecoder();x=[torch.randn(1,n,d) for n,d in [(512,3),(512,3),(32,3),(32,3),(128,16),(64,16)]]
target=torch.rand(1,512,32).softmax(-1)
start=time.perf_counter();w=model(*x);loss=soft_cross_entropy(w,target);loss.backward();cpu_s=time.perf_counter()-start
result=dict(scope='Untrained components only: no DINO/TRELLIS latent extraction, no generator training, no original benchmark',seed=2026,
    quantization=rows,torch=torch.__version__,platform=platform.platform(),width=32,points=512,bones=32,surface_tokens=128,skeleton_tokens=64,
    cpu_forward_backward_s=cpu_s,probability_sum_error=float((w.sum(-1)-1).abs().max().detach()),parameters=sum(p.numel() for p in model.parameters()))
if args.mps:
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
    import copy
    other=copy.deepcopy(model).to('mps');other.zero_grad();start=time.perf_counter()
    wm=other(*[v.to('mps') for v in x]);lm=soft_cross_entropy(wm,target.to('mps'));lm.backward();torch.mps.synchronize()
    result.update(mps_forward_backward_s=time.perf_counter()-start,cpu_mps_output_max_diff=float((w.detach()-wm.cpu().detach()).abs().max()),
        cpu_mps_gradient_max_diff=max(float((a.grad-b.grad.cpu()).abs().max()) for a,b in zip(model.parameters(),other.parameters())))
(root/'results/component_check.json').write_text(json.dumps(result,indent=2)+'\n')
# One procedural tree: visualization of the actual closed-cell voxelizer.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
j=np.array([[0,-.35,0],[0,0,0],[-.3,.25,.1],[.3,.25,.1],[0,.4,0]])
p=np.array([0,0,1,1,1]);cells=np.unique(np.concatenate([segment_voxels(j[p[i]],j[i],64) for i in range(1,len(j))]),axis=0)
fig=plt.figure(figsize=(9,4),layout='constrained');ax=fig.add_subplot(121,projection='3d')
c=(cells+.5)/64-.5;ax.scatter(*c.T,s=5,alpha=.4,color='#0072B2')
for i in range(1,len(j)):ax.plot(*j[[p[i],i]].T,color='#D55E00',lw=2)
ax.set(xlabel='x',ylabel='y',zlabel='z');ax.set_box_aspect([1,1,1])
ax=fig.add_subplot(122);ax.plot([r['resolution'] for r in rows],[r['max_error'] for r in rows],'o-',label='Measured maximum')
ax.plot([r['resolution'] for r in rows],[r['bound'] for r in rows],'--',label='Rounding bound');ax.set(xlabel='Joint coordinate grid resolution',ylabel='Quantization error (unit bbox)');ax.legend(fontsize=8)
fig.savefig(root/'results/representation_check.png',dpi=160);plt.close(fig);print(json.dumps(result))
