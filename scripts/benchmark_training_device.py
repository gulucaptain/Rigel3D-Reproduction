"""Synthetic shape-matched timing audit; not an accuracy experiment."""
import json,time
from pathlib import Path
import torch
from rigel3d.decoders import RigDecoders,skeleton_loss
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);rows=[]
for device in ['cpu','mps']:
    torch.manual_seed(0);m=RigDecoders().to(device);opt=torch.optim.AdamW(m.parameters());b=4
    sf=torch.randn(b,48,16,device=device);sc=torch.randn(b,48,3,device=device);rf=torch.randn_like(sf);rc=torch.randn_like(sc);p=torch.randn(b,256,3,device=device);n=torch.randn_like(p);h=torch.randn(b,32,3,device=device);t=torch.randn_like(h);tokens=torch.zeros(b,37,4,device=device,dtype=torch.long);lens=torch.full((b,),37,device=device,dtype=torch.long);start=time.perf_counter()
    for step in range(60):
        opt.zero_grad();s,r=m.encoder(sf,sc,rf,rc);o=m.skeleton(tokens,s,r);sk,_=skeleton_loss(o,tokens,lens,True);w=m.skinning(p,n,h,t,s,r);loss=sk-w.clamp_min(1e-12).log().mean();loss.backward();ok=torch.isfinite(torch.cat([v.grad.flatten() for v in m.parameters() if v.grad is not None])).all()
        if not ok:raise ValueError('grad')
        opt.step()
        if step==9:
            if device=='mps':torch.mps.synchronize()
            start=time.perf_counter()
    if device=='mps':torch.mps.synchronize()
    rows.append(dict(device=device,seconds_per_step=(time.perf_counter()-start)/50));print(rows[-1],flush=True)
(root/'results/round2/device_timing.json').write_text(json.dumps(dict(scope='Synthetic timing only, 4x256 points,32 bones,37 joints; finite gradient cat check; 10 warmup+50 timed steps',checks=rows),indent=2)+'\n')
