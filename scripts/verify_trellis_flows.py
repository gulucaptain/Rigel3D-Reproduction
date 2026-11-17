"""Independent CPU/MPS velocity replay for each selected small flow model."""
import json,hashlib
from pathlib import Path
import torch
from rigel3d.trellis_flow import CoupledFlows
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);checks=[];torch.manual_seed(91);state=torch.randn(1,16,16,16,16);image=torch.randn(1,64,1024);t=torch.tensor([.4]);coords=torch.tensor([[-.3,.2,.1],[.25,-.12,.31]]);z=torch.randn(2,8)
for run in ['flows_frozen_surface','flows_balanced_rig','flows_full_joint','flows_full_joint_balanced']:
    for seed in [0,1,2]:
        p=root/f'results/round3/{run}/seed{seed}/best.pt';assert p.exists();weights=torch.load(p,weights_only=True);outputs=[]
        for device in ['cpu','mps']:
            m=CoupledFlows().to(device).eval();m.load_state_dict(weights)
            with torch.no_grad():out=m.velocity(state.to(device),t.to(device),image.to(device));local=m.local['rig'](z.to(device),t.to(device),image.to(device),coords.to(device),state.to(device));outputs.append((out.cpu(),local.cpu()))
            del m
        delta=max(float((a-b).abs().max()) for a,b in zip(outputs[0],outputs[1]));assert delta<2e-4,(run,seed,delta);checks.append(dict(run=run,seed=seed,passed=True,max_cpu_mps_difference=delta,checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(root/'results/round3/flow_replay.json').write_text(json.dumps(dict(passed=True,scope='Fixed normalized noisy occupancy state, image tokens and2 local rig queries per selected checkpoint. CPU/MPS velocity outputs, not exhaustive optimization replay.',checks=checks),indent=2)+'\n');print(json.dumps(dict(passed=True,models=len(checks))))
