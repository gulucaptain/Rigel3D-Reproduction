"""Single-view image conditioning: DINO-L patch tokens pooled37x37 ->8x8."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from PIL import Image
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.hub.set_dir(str(root/'data/external/torchhub'));rev=json.load(open(root/'sources/dinov2-revision.json'))['commit'];m=torch.hub.load('facebookresearch/dinov2:'+rev,'dinov2_vitl14_reg',pretrained=True,trust_repo=True).to('mps').eval();rows=[]
for r in json.load(open(root/'results/round3/dino_features.json'))['assets']:
    base=root/'data/external/dino_cohort'/Path(r['filename']).stem;dest=base/'trellis_image_tokens.npz'
    if not dest.exists():
        tokens=[]
        with torch.inference_mode():
            for i in range(4):
                image=np.asarray(Image.open(base/f'view_{i}.png').convert('RGB').resize((518,518),Image.Resampling.LANCZOS));x=torch.tensor(image.transpose(2,0,1).copy(),device='mps',dtype=torch.float32)[None]/255;x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
                p=m.forward_features(x)['x_prenorm'][:,1+m.num_register_tokens:];p=torch.nn.functional.layer_norm(p,(1024,));p=p.reshape(1,37,37,1024).permute(0,3,1,2);p=torch.nn.functional.adaptive_avg_pool2d(p.cpu(),(8,8)).flatten(2).transpose(1,2);tokens.append(p[0].cpu().numpy().astype(np.float16))
        np.savez_compressed(dest,tokens=np.stack(tokens))
    rows.append(dict(filename=r['filename'],sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print(r['filename'],flush=True)
(root/'results/round3/image_tokens.json').write_text(json.dumps(dict(scope='Real single-view DINO-L normalized patch tokens, spatial8x8 pooling for Mac flow training. Native224 renders upsampled518; pooled64 tokens instead of original1374-token condition. No GT input at inference.',assets=rows),indent=2)+'\n')
