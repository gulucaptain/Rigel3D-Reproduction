"""Private qualitative panels: first four fixed test assets, seed0, no selection."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
root=Path(__file__).resolve().parents[1];records=[r for r in json.load(open(root/'results/round2/dino_features.json'))['assets'] if r['split']=='test'][:4];fig=plt.figure(figsize=(10,11),layout='constrained')
for i,r in enumerate(records):
    base=root/'data/external/dino_cohort'/Path(r['filename']).stem;ax=fig.add_subplot(4,3,3*i+1);ax.imshow(Image.open(base/'view_0.png'));ax.axis('off')
    if i==0:ax.set_title('Single input image')
    for j,key in enumerate(['reconstruction_','']):
        ax=fig.add_subplot(4,3,3*i+j+2,projection='3d');path=root/'data/external/generated/seed0'/(key+r['filename'])
        if path.exists():
            d=np.load(path);v=d['vertices'];f=d['faces'];p=v[:,[0,2,1]].copy();p[:,1]*=-1
            if len(f):ax.add_collection3d(Poly3DCollection(p[f],facecolor='#7299bd',edgecolor='none',alpha=.55))
            if key=='':
                joints=d['joints'][:,[0,2,1]].copy();joints[:,1]*=-1
                for n,parent in enumerate(d['parents']):
                    a=joints[[parent,n]];ax.plot(a[:,0],a[:,1],a[:,2],c='#c54030',lw=1)
        ax.set(xlim=(-.55,.55),ylim=(-.55,.55),zlim=(-.55,.55));ax.set_box_aspect([1,1,1]);ax.view_init(18,-55);ax.axis('off')
        if i==0:ax.set_title('GT-latent reconstruction' if key else 'Image-only generation + skeleton')
folder=root/'data/external/qualitative';folder.mkdir(exist_ok=True);fig.savefig(folder/'round2_fixed_test_seed0.png',dpi=150);plt.close(fig)
(folder/'caption.json').write_text(json.dumps(dict(seed=0,selection='First four fixed test-cohort records; no selection by result quality',assets=[r['filename'] for r in records],scope='Middle uses GT sparse positions and GT rig/skin encoded latents. Right uses single-view DINO + noise only. Small independent models; not paper quality. Raw-derived views remain local.'),indent=2)+'\n')
