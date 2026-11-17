"""Large local checkpoint hashes, actual backbone updates, CPU/MPS output replay."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from rigel3d.joint_trellis import FullJointTrellis
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);original_rig=load_file(str(root/'data/external/trellis/weights/slat_enc_swin8_B_64l8_fp16.safetensors'));original_mesh=load_file(str(root/'data/external/trellis/weights/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors'));r=next(r for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets'] if r['split']=='val');base=root/'data/external/dino_cohort'/Path(r['filename']).stem;d=np.load(base/'trellis_latents.npz');rf=np.load(base/'trellis_features.npz')['rig_features'].astype('float32');checks=[]
for seed in [0,1,2]:
    folder=root/f'results/round3/full_joint/seed{seed}';manifest=json.load(open(folder/'model_manifest.json'));path=root/manifest['checkpoint'];assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256'];state=torch.load(path,weights_only=True);changes={}
    for prefix,reference in [('rig_encoder',original_rig),('mesh',original_mesh)]:
        for block in [0,11]:
            key=f'blocks.{block}.attn.to_qkv.weight';delta=float((state[prefix+'.'+key]-reference[key].float()).abs().max());assert delta>0;changes[f'{prefix}_block{block}_max_update']=delta
    for k in ['weight','bias']:assert torch.equal(state['surface_head.'+k],original_rig['out_layer.'+k])
    outputs=[]
    for device in ['cpu','mps']:
        model=FullJointTrellis(root/'data/external/trellis/weights',device=device).eval();model.load_state_dict(state,strict=True)
        with torch.no_grad():
            rz,mu,lv=model.encode_rig(torch.tensor(rf,device=device),d['rig_coords'],False);rc=torch.tensor(((d['rig_coords']+.5)/64-.5).astype('float32'),device=device);fields=model.mesh_fields(torch.tensor(d['surface_mu'][:8],device=device),d['surface_coords'][:8],mu[:48],rc[:48]);outputs.append((mu.cpu(),fields.feats.cpu()))
        del model
        if device=='mps':torch.mps.empty_cache()
    rig_error=float((outputs[0][0]-outputs[1][0]).abs().max());mesh_error=float((outputs[0][1]-outputs[1][1]).abs().max());assert rig_error<1e-3 and mesh_error<1e-3,(seed,rig_error,mesh_error);checks.append(dict(seed=seed,passed=True,surface_head_bitwise_frozen=True,rig_max_cpu_mps_error=rig_error,mesh_fields_max_cpu_mps_error=mesh_error,checkpoint_sha256=manifest['sha256'],**changes));del state
(root/'results/round3/full_joint_replay.json').write_text(json.dumps(dict(passed=True,scope='All3 large checkpoint hashes; first/last actual rig+mesh transformer weight updates; frozen surface head; CPU/MPS whole rig encoder and8-cube full mesh field replay on one validation asset. Not CUDA rendering parity.',checks=checks),indent=2)+'\n');print(json.dumps(checks))
