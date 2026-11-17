"""GT-latent perturbation audit on first3 validation assets, not generation."""
import json
from pathlib import Path
import numpy as np,torch
from rigel3d.trellis_runtime import load_occupancy_decoder
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);m=load_occupancy_decoder(root/'data/external/trellis/weights');records=[r for r in json.load(open(root/'results/round3/pretrained_latents.json'))['assets'] if r['split']=='val'][:3];stats=np.load(root/'results/round3/flows_frozen_surface/seed0/stats.npz');rows=[]
with torch.no_grad():
    for r in records:
        base=root/'data/external/dino_cohort'/Path(r['filename']).stem;source=np.load(base/'features.npz');cache=np.load(base/'occupancy_pretrained.npz')
        for mode in ['surface','rig']:
            truth=np.zeros((64,64,64),bool);coords=source[mode+'_coords'];truth[coords[:,0],coords[:,1],coords[:,2]]=True;mu=torch.tensor(cache[mode+'_mu'][None],device='mps');std=torch.tensor(stats[mode+'_occ_std'],device='mps')[None,:,None,None,None]
            for sigma,seeds in [(0,[0]),(.1,[0,1,2]),(.5,[0,1,2])]:
                for seed in seeds:
                    torch.manual_seed(seed);z=mu+sigma*std*torch.randn_like(mu);pred=(m(z)[0,0]>0).cpu().numpy();inter=(pred&truth).sum();union=(pred|truth).sum();rows.append(dict(filename=r['filename'],mode=mode,sigma=sigma,seed=seed,iou=float(inter/max(union,1)),recall=float(inter/max(truth.sum(),1)),predicted_voxels=int(pred.sum())));print(r['filename'],mode,sigma,seed,rows[-1]['iou'],flush=True)
(root/'results/round3/occupancy_robustness.json').write_text(json.dumps(dict(scope='First3 fixed VALIDATION assets,3 perturbation seeds for each positive noise level. Perturb GT occupancy posterior by sigma times train-channel std, then decode with frozen official VAE. This is a GT perturbation diagnostic, not image generation.',assets=rows),indent=2)+'\n')
