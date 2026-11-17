"""Real pretrained mesh loss backward reaches posterior head and new rig CA."""
import json
from pathlib import Path
import numpy as np,torch
from rigel3d.joint_trellis import JointTrellis
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.manual_seed(31);r=next(r for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets'] if r['split']=='train');d=np.load(root/'data/external/dino_cohort'/Path(r['filename']).stem/'trellis_latents.npz');m=JointTrellis(root/'data/external/trellis/weights');m.surface_head.requires_grad_(False);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4);rows=[]
for step in [0,1]:
    opt.zero_grad();sf,_,_,_=m.posterior(torch.tensor(d['surface_hidden'][:8].astype('float32'),device='mps'),'surface');rf,_,_,_=m.posterior(torch.tensor(d['rig_hidden'][:8].astype('float32'),device='mps'),'rig');rc=torch.tensor(((d['rig_coords'][:8]+.5)/64-.5).astype('float32'),device='mps');f=m.mesh_fields(sf,d['surface_coords'][:8],rf,rc);loss=f.feats[:,:8].square().mean();loss.backward();row=dict(step=step,loss=float(loss.detach().cpu()))
    for key in ['surface_head','rig_head','mesh_adapter']:
        grads=[p.grad for p in getattr(m,key).parameters() if p.grad is not None];row[key+'_grad_l1']=float(sum(g.abs().sum() for g in grads).cpu()) if grads else 0
    row['frozen_mesh_has_grad']=any(p.grad is not None for p in m.mesh.parameters());assert row['surface_head_grad_l1']==0 and row['mesh_adapter_grad_l1']>0 and not row['frozen_mesh_has_grad']
    if step==1:assert row['rig_head_grad_l1']>0
    opt.step();rows.append(row)
(root/'results/round3/joint_gradient_audit.json').write_text(json.dumps(dict(scope='Mesh output SDF-only loss, no AR/skin terms; surface encoder stays frozen; gradients reach mesh rig CA; rig-head gradient activates after initial zero CA output projection updates. Mesh backbone frozen.',checks=rows),indent=2)+'\n');print(json.dumps(rows))
