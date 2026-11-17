"""Generate from one PNG/JPEG. No dataset/GT loader is used by this entry point."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageOps
from rigel3d.decoders import RigDecoders
from rigel3d.generation import FourFlows,FieldDecoder
from rigel3d.inference import generate_asset
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--image',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--noise-seed',type=int,default=0);ap.add_argument('--device',default='mps');a=ap.parse_args();torch.set_num_threads(4)
folder=root/f'results/round2/generation/seed{a.seed}';flows=FourFlows().to(a.device).eval();field=FieldDecoder().to(a.device).eval();dec=RigDecoders(channels=8).to(a.device).eval()
for m,p in [(flows,folder/'flows.pt'),(field,folder/'field.pt'),(dec,root/f'results/round2_vae/rig/seed{a.seed}/best.pt')]:m.load_state_dict(torch.load(p,weights_only=True))
torch.hub.set_dir(str(root/'data/external/torchhub'));rev=json.loads((root/'sources/dinov2-revision.json').read_text())['commit'];dino=torch.hub.load('facebookresearch/dinov2:'+rev,'dinov2_vits14',pretrained=True,trust_repo=True).eval().to(a.device)
image=ImageOps.fit(Image.open(a.image).convert('RGB'),(224,224),method=Image.Resampling.BICUBIC);x=torch.tensor(np.asarray(image).transpose(2,0,1).copy(),device=a.device,dtype=torch.float32)[None]/255
x=(x-torch.tensor([.485,.456,.406],device=a.device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=a.device)[None,:,None,None]
with torch.inference_mode():condition=dino.forward_features(x)['x_norm_patchtokens'].mean(1)
torch.manual_seed(a.noise_seed);pred,diag=generate_asset(condition,flows,field,dec,dict(np.load(folder/'latent_stats.npz')))
a.output.mkdir(parents=True,exist_ok=False);(a.output/'status.json').write_text(json.dumps(diag,indent=2)+'\n')
if pred is not None:
    np.savez_compressed(a.output/'asset.npz',**pred)
    with (a.output/'mesh.obj').open('w') as f:
        for v in pred['vertices']:f.write('v '+' '.join(map(str,v))+'\n')
        for face in pred['faces']:f.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    (a.output/'skeleton.json').write_text(json.dumps(dict(joints=pred['joints'].tolist(),parents=pred['parents'].tolist()),indent=2)+'\n')
print(json.dumps(diag))
