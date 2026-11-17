"""Record real upstream FlexiCubes CPU and native MPS numerical/gradient checks."""
import hashlib,json
from pathlib import Path
import torch
from rigel3d.flexicubes import regular_grid,FlexiCubes
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);rows=[]
for device in ['cpu','mps']:
    try:
        q,c=regular_grid(12,device);s=(q.norm(dim=-1)-.3).requires_grad_();v,f,reg,_=FlexiCubes(device)(q,s,c,12,training=True);loss=v.square().mean()+reg.mean();loss.backward()
        rows.append(dict(device=device,success=True,vertices=len(v),faces=len(f),radius_mean=float(v.norm(dim=-1).mean().detach().cpu()),finite_grad=bool(torch.isfinite(s.grad).all()),grad_l1=float(s.grad.abs().sum().cpu())))
    except Exception as e:rows.append(dict(device=device,success=False,error=type(e).__name__+': '+str(e)))
report=dict(upstream='https://github.com/MaxtirError/FlexiCubes',commit='815e075a2a400d06c48d94c347674344ed6ae5c5',license='Apache-2.0',modification='Replace only kaolin.utils.testing.check_tensor with local shape check; geometry code unchanged',source_sha256=hashlib.sha256((root/'sources/flexicubes.py.txt').read_bytes()).hexdigest(),checks=rows)
(root/'results/round2/flexicubes_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
