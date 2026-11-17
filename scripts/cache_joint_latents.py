"""Cache selected joint posterior means, keeping GT-coordinate reconstruction explicit."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--joint-run',default='joint_frozen_surface');a=ap.parse_args();torch.set_num_threads(4);records=json.load(open(root/'results/round3/pretrained_latents.json'))['assets'];rows=[]
for seed in [0,1,2]:
    folder=root/f'results/round3/{a.joint_run}/seed{seed}';assert (folder/'training.json').exists(),'Training not complete';state=torch.load(folder/'best.pt',weights_only=True);dest=root/f'data/external/round3_{a.joint_run}_latents/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    heads={}
    for mode in ['surface','rig']:
        h=torch.nn.Linear(768,16);h.load_state_dict({k:state[mode+'_head.'+k] for k in ['weight','bias']});heads[mode]=h.eval()
    for r in records:
        base=root/'data/external/dino_cohort'/Path(r['filename']).stem;d=np.load(base/'trellis_latents.npz');out={}
        with torch.no_grad():
            for mode,h in heads.items():
                mu,lv=h(torch.tensor(d[mode+'_hidden'].astype('float32'))).chunk(2,-1);out[mode+'_mu']=mu.numpy();out[mode+'_logvar']=lv.numpy();out[mode+'_coords']=d[mode+'_coords']
        p=dest/r['filename'];np.savez_compressed(p,**out);rows.append(dict(seed=seed,filename=r['filename'],split=r['split'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(root/f'results/round3/{a.joint_run}_latents.json').write_text(json.dumps(dict(scope='Validation-selected joint adapted posterior means from GT features/coordinates; generator training targets only, not generation inputs.',assets=rows),indent=2)+'\n')
