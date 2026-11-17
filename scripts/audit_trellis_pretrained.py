"""Load every official key; CPU/MPS encoder parity, native mesh smoke and timings."""
import json,time
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from rigel3d.trellis_native import Sparse,load_pretrained
from rigel3d.vendor.trellis.occupancy import SparseStructureEncoder,SparseStructureDecoder
root=Path(__file__).resolve().parents[1];base=root/'data/external/trellis/weights';torch.set_num_threads(4);rows=[];torch.manual_seed(44)
coords=np.array([[0,0,0],[4,3,2],[7,7,7],[8,8,8],[20,22,21]],np.int32);features=torch.randn(5,1024);reference=None
for device in ['cpu','mps']:
    start=time.perf_counter();m=load_pretrained('encoder',base,device).eval()
    with torch.no_grad():z,mu,lv=m(Sparse(features.to(device),coords));value=mu.cpu()
    delta=0 if reference is None else float((value-reference).abs().max());reference=value if reference is None else reference
    rows.append(dict(component='surface_encoder',device=device,strict_keys=len(m.state_dict()),parameters=sum(p.numel() for p in m.parameters()),finite=bool(torch.isfinite(value).all()),max_cpu_difference=delta,elapsed_s=time.perf_counter()-start));del m
m=load_pretrained('mesh',base,'mps').eval();start=time.perf_counter()
with torch.no_grad():out=m(Sparse(torch.randn(5,8,device='mps'),coords))
rows.append(dict(component='mesh_decoder',device='mps',strict_keys=len(m.state_dict()),parameters=sum(p.numel() for p in m.parameters()),output_cubes=len(out.coords),output_channels=out.feats.shape[-1],finite=bool(torch.isfinite(out.feats).all()),elapsed_s=time.perf_counter()-start));del m
for mode,cls in [('enc',SparseStructureEncoder),('dec',SparseStructureDecoder)]:
    name='ss_'+mode+'_conv3d_16l8_fp16';config=json.load(open(base/(name+'.json')));m=cls(**config['args']);m.load_state_dict(load_file(str(base/(name+'.safetensors'))),strict=True,assign=True);rows.append(dict(component='occupancy_'+mode,strict_keys=len(m.state_dict()),parameters=sum(p.numel() for p in m.parameters())));del m
(root/'results/round3/pretrained_kernel_audit.json').write_text(json.dumps(dict(source='microsoft/TRELLIS @193e0502b5cb3eb5997ec457d5340e95d09b158e',checks=rows,limits='Native sparse kernels verified against dense/window reference tests; no CUDA backend available for complete original-vs-native parity. Mesh geometry extraction and actual data evaluated separately.'),indent=2)+'\n');print(json.dumps(rows))
