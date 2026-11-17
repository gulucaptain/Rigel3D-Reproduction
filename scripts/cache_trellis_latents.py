"""Cache strict official pretrained encoder posteriors and frozen hidden states."""
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
from rigel3d.trellis_native import Sparse,load_pretrained
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);model=load_pretrained('encoder',root/'data/external/trellis/weights','mps').float().eval();rows=[]
for r in json.load(open(root/'results/round3/dino_features.json'))['assets']:
    base=root/'data/external/dino_cohort'/Path(r['filename']).stem;d=np.load(base/'trellis_features.npz');dest=base/'trellis_latents.npz';start=time.perf_counter();out={}
    if dest.exists():rows.append(json.load(open(base/'trellis_latents.json')));continue
    with torch.no_grad():
        for mode in ['surface','rig']:
            coords=d[mode+'_coords'];x=Sparse(torch.tensor(d[mode+'_features'].astype('float32'),device='mps'),coords);h=model.trunk(x);h=torch.nn.functional.layer_norm(h.feats.float(),(768,));stats=model.out_layer(h);mu,lv=stats.chunk(2,-1)
            out[mode+'_coords']=coords;out[mode+'_hidden']=h.cpu().numpy().astype(np.float16);out[mode+'_mu']=mu.cpu().numpy();out[mode+'_logvar']=lv.cpu().numpy()
    out['image_features']=d['image_features'];np.savez_compressed(dest,**out);row=dict(filename=r['filename'],split=r['split'],elapsed_s=time.perf_counter()-start,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),surface_count=len(out['surface_coords']),rig_count=len(out['rig_coords']));rows.append(row);(base/'trellis_latents.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
(root/'results/round3/pretrained_latents.json').write_text(json.dumps(dict(scope='GT active coordinates; official pretrained encoder for surface and same initialization for rig; FP32 compute. Frozen normalized768 hidden cache for explicit head/adapter joint tuning, not full backbone finetuning.',assets=rows),indent=2)+'\n')
