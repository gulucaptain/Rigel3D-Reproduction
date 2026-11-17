"""Verify source hashes, frozen surface weights, deterministic CPU/MPS AR replay."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file
from rigel3d.decoders import RigDecoders
from rigel3d.data import fps_indices
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);source=json.load(open(root/'results/round3/pretrained_sources.json'));checks=[]
for entry in source['files']:
    p=root/'data/external/trellis/weights'/Path(entry['filename']).name;assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
original=load_file(str(root/'data/external/trellis/weights/slat_enc_swin8_B_64l8_fp16.safetensors'));record=next(r for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets'] if r['split']=='test');base=root/'data/external/dino_cohort'/Path(record['filename']).stem;d=np.load(base/'trellis_latents.npz')
for seed in [0,1,2]:
    folder=root/f'results/round3/joint_frozen_surface/seed{seed}';state=torch.load(folder/'best.pt',weights_only=True)
    for k in ['weight','bias']:assert torch.equal(state['surface_head.'+k],original['out_layer.'+k])
    cpu=RigDecoders(channels=8).eval();cpu.load_state_dict({k.removeprefix('rig_decoders.'):v for k,v in state.items() if k.startswith('rig_decoders.')});features=[]
    for mode in ['surface','rig']:
        head=torch.nn.Linear(768,16);head.load_state_dict({k:state[mode+'_head.'+k] for k in ['weight','bias']});idx=fps_indices(d[mode+'_coords'],48)
        with torch.no_grad():mu=head(torch.tensor(d[mode+'_hidden'][idx].astype('float32')))[:,:8]
        features.extend([mu[None],torch.tensor(((d[mode+'_coords'][idx]+.5)/64-.5).astype('float32'))[None]])
    with torch.no_grad():
        s,r=cpu.encoder(*features);tokens=torch.tensor([[[0,0,0,0],[32,64,96,0]]]);out=cpu.skeleton(tokens,s,r);gpu=RigDecoders(channels=8).to('mps').eval();gpu.load_state_dict(cpu.state_dict());gs,gr=gpu.encoder(*[v.to('mps') for v in features]);g=gpu.skeleton(tokens.to('mps'),gs,gr);delta=max(float((out[k]-g[k].cpu()).abs().max()) for k in out)
    assert delta<2e-4,(seed,delta);checks.append(dict(seed=seed,surface_head_bitwise_frozen=True,cpu_mps_logits_max_abs_difference=delta,checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),passed=True))
(root/'results/round3/replay_audit.json').write_text(json.dumps(dict(passed=True,official_weight_hashes=len(source['files']),checks=checks,scope='Source hashes and frozen surface parameters checked. CPU/MPS AR logits replay on one fixed held-out asset and two-token probe per seed; not full CUDA mesh parity or exhaustive optimization replay.'),indent=2)+'\n');print(json.dumps(checks))
