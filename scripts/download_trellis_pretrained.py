"""Pinned official public TRELLIS VAE checkpoints; no credentials, verify LFS hashes."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1];manifest=json.load(open(root/'sources/trellis/hf-manifest.json'));revision=manifest['sha'];base=root/'data/external/trellis/weights';base.mkdir(parents=True,exist_ok=True)
names=['slat_enc_swin8_B_64l8_fp16','slat_dec_mesh_swin8_B_64l8m256c_fp16','ss_enc_conv3d_16l8_fp16','ss_dec_conv3d_16l8_fp16'];records=[]
def fetch(entry):
    name=entry['rfilename'];path=base/Path(name).name;url=f'https://huggingface.co/microsoft/TRELLIS-image-large/resolve/{revision}/{name}'
    expected=entry.get('lfs',{}).get('sha256')
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(url,timeout=90) as src,tmp.open('wb') as dst:
            while chunk:=src.read(8*1024*1024):dst.write(chunk)
        tmp.rename(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and expected!=digest:raise ValueError('Checkpoint hash mismatch '+name)
    rec=dict(filename=name,bytes=path.stat().st_size,sha256=digest,url=url);print(json.dumps(rec),flush=True);return rec
entries=[r for r in manifest['siblings'] if any(r['rfilename']=='ckpts/'+n+ext for n in names for ext in ['.json','.safetensors'])]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,entries))
out=root/'results/round3';out.mkdir(exist_ok=True);(out/'pretrained_sources.json').write_text(json.dumps(dict(repository='microsoft/TRELLIS-image-large',revision=revision,files=records),indent=2)+'\n')
