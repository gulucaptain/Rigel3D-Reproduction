"""Standalone image-only entry: no dataset/GT files opened."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
from rigel3d.joint_trellis import JointTrellis,FullJointTrellis
from rigel3d.trellis_flow import CoupledFlows
from rigel3d.trellis_runtime import generate_image_only,load_occupancy_decoder
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--image',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--noise-seed',type=int,default=123);ap.add_argument('--cfg',type=float,default=5);ap.add_argument('--full-joint',action='store_true');ap.add_argument('--joint-run',default='joint_frozen_surface');ap.add_argument('--flow-run',default='flows_frozen_surface');a=ap.parse_args();torch.set_num_threads(4);folder=root/f'results/round3/{a.flow_run}/seed{a.seed}';flows=CoupledFlows().to('mps').eval();flows.load_state_dict(torch.load(folder/'best.pt',weights_only=True));joint=(FullJointTrellis if a.full_joint else JointTrellis)(root/'data/external/trellis/weights').eval()
if a.full_joint:
    manifest=json.load(open(root/f'results/round3/{a.joint_run}/seed{a.seed}/model_manifest.json'));joint.load_state_dict(torch.load(root/manifest['checkpoint'],weights_only=True),strict=True)
else:joint.load_trainable_state(torch.load(root/f'results/round3/{a.joint_run}/seed{a.seed}/best.pt',weights_only=True))
decoder=load_occupancy_decoder(root/'data/external/trellis/weights')
torch.hub.set_dir(str(root/'data/external/torchhub'));rev=json.load(open(root/'sources/dinov2-revision.json'))['commit'];m=torch.hub.load('facebookresearch/dinov2:'+rev,'dinov2_vitl14_reg',pretrained=True,trust_repo=True).eval().to('mps');image=np.asarray(Image.open(a.image).convert('RGB').resize((518,518),Image.Resampling.LANCZOS));x=torch.tensor(image.transpose(2,0,1).copy(),device='mps',dtype=torch.float32)[None]/255;x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
with torch.inference_mode():
    p=m.forward_features(x)['x_prenorm'][:,1+m.num_register_tokens:];p=torch.nn.functional.layer_norm(p,(1024,));p=p.reshape(1,37,37,1024).permute(0,3,1,2).cpu();condition=torch.nn.functional.adaptive_avg_pool2d(p,(8,8)).flatten(2).transpose(1,2).half().float().to('mps')
del m;torch.mps.empty_cache();torch.manual_seed(a.noise_seed);pred,diag=generate_image_only(condition,flows,joint,decoder,dict(np.load(folder/'stats.npz')),cfg=a.cfg);a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'asset.npz',**pred);(a.output/'status.json').write_text(json.dumps(diag,indent=2)+'\n')
if 'vertices' in pred:
    with (a.output/'mesh.obj').open('w') as f:
        for v in pred['vertices']:f.write('v '+' '.join(map(str,v))+'\n')
        for face in pred['faces']:f.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    (a.output/'skeleton.json').write_text(json.dumps(dict(joints=pred['joints'].tolist(),parents=pred['parents'].tolist()),indent=2)+'\n')
print(json.dumps(diag))
