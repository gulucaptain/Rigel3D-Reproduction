"""Official 64^3 -> 16^3x8 ->64^3 occupancy VAE, actual surface/rig targets."""
import json,time
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from rigel3d.vendor.trellis.occupancy import SparseStructureEncoder,SparseStructureDecoder
root=Path(__file__).resolve().parents[1];base=root/'data/external/trellis/weights';torch.set_num_threads(4)
def load(mode,cls):
    name='ss_'+mode+'_conv3d_16l8_fp16';config=json.load(open(base/(name+'.json')));m=cls(**config['args']);m.load_state_dict(load_file(str(base/(name+'.safetensors'))),strict=True,assign=True);m.convert_to_fp32();return m.to('mps').eval()
e=load('enc',SparseStructureEncoder);d=load('dec',SparseStructureDecoder);rows=[]
for r in json.load(open(root/'results/round2/dino_features.json'))['assets']:
    base=root/'data/external/dino_cohort'/Path(r['filename']).stem;source=np.load(base/'features.npz');item={};row=dict(filename=r['filename'],split=r['split']);start=time.perf_counter()
    with torch.no_grad():
        for mode in ['surface','rig']:
            coords=source[mode+'_coords'];occ=torch.zeros(1,1,64,64,64,device='mps');ix=torch.tensor(coords,device='mps').long();occ[0,0,ix[:,0],ix[:,1],ix[:,2]]=1
            z,mu,lv=e(occ,False,True);logits=d(z);p=logits>0;t=occ>0;inter=(p&t).sum();union=(p|t).sum()
            row[mode]=dict(iou=float((inter/union.clamp_min(1)).cpu()),precision=float((inter/p.sum().clamp_min(1)).cpu()),recall=float((inter/t.sum().clamp_min(1)).cpu()),predicted_voxels=int(p.sum().cpu()),target_voxels=len(coords),latent_shape=list(mu.shape),bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits,occ).cpu()))
            item[mode+'_mu']=mu[0].cpu().numpy();item[mode+'_logvar']=lv[0].cpu().numpy();item[mode+'_base_logits']=logits[0,0].cpu().numpy().astype(np.float16)
    np.savez_compressed(base/'occupancy_pretrained.npz',**item);row['elapsed_s']=time.perf_counter()-start;rows.append(row);print(json.dumps(row),flush=True)
(root/'results/round3/occupancy_pretrained.json').write_text(json.dumps(dict(scope='GT binary64^3 occupancy encoded into continuous16^3x8 posterior mean and decoded. Not generated occupancy. Official surface pretrained VAE also applied to thin rig as transfer baseline.',assets=rows),indent=2)+'\n')
