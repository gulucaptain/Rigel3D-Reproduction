"""Separate fp16 drift from structural port errors using FP32 CPU/MPS replay."""
import json
from pathlib import Path
import numpy as np,torch
from rigel3d.trellis_native import Sparse,load_pretrained
root=Path(__file__).resolve().parents[1];torch.manual_seed(44);x=torch.randn(32,1024);coords=np.random.default_rng(44).integers(0,24,(32,3));results=[];reference=None
for device in ['cpu','mps']:
    m=load_pretrained('encoder',root/'data/external/trellis/weights',device).float().eval()
    with torch.no_grad():value=m(Sparse(x.to(device),coords))[1].cpu()
    if reference is None:reference=value
    delta=float((value-reference).abs().max());results.append(dict(device=device,max_abs_cpu_difference=delta,latent_rms=float(value.square().mean().sqrt())))
    assert delta<5e-4,(device,delta)
(root/'results/round3/precision_audit.json').write_text(json.dumps(dict(checks=results,choice='FP32 compute for actual pretrained encoding/mesh and joint adapters; checkpoint values unchanged apart from lossless fp16->fp32 expansion. Mixed fp16 CPU/MPS drift was .0179 on5-token probe, retained separately.'),indent=2)+'\n');print(json.dumps(results))
