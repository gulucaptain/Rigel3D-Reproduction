"""Trace subset hashes to original meshes using fixed official metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
root=Path(__file__).resolve().parents[1];base=root/'data/external';base.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--download-index',action='store_true');args=ap.parse_args()
sha='dc96b5abb010c7fc7834335efad64c3a2e8a47a4';meta=base/'Anymate_id.json'
url=f'https://raw.githubusercontent.com/yfde/Anymate/{sha}/Anymate/data/Anymate_id.json'
if not meta.exists():
    with urllib.request.urlopen(url,timeout=60) as r:meta.write_bytes(r.read(32*1024*1024))
rows=json.loads(meta.read_bytes());subset=json.loads((root/'results/anymate_subset_manifest.json').read_text())
families={'/'.join(r['asset_name'].split('/')[:2]) for r in subset['assets'] if r['split']!='test'}
mapping=[r for r in rows if r['name'] in families]
index=base/'objaverse-github.parquet';indexurl='https://huggingface.co/datasets/allenai/objaverse-xl/resolve/ee4fa27ac534725e24fd1a73c0dcdaccdee75819/github/github.parquet'
if not index.exists():
    if not args.download_index:raise FileNotFoundError('Use --download-index for the 797MB public metadata index')
    total=0
    with urllib.request.urlopen(indexurl,timeout=60) as r,index.open('wb') as f:
        while chunk:=r.read(8*1024*1024):
            total+=len(chunk)
            if total>797466051:raise ValueError('Index size exceeded')
            f.write(chunk)
if index.stat().st_size!=797466051:raise ValueError('Truncated metadata index')
wanted={r['id'] for r in mapping};locations=[]
for b in pq.ParquetFile(index).iter_batches(batch_size=65536,columns=['fileIdentifier','sha256','license','fileType']):
    t=pa.Table.from_batches([b]);locations.extend(t.filter(pc.is_in(t['sha256'],value_set=pa.array(sorted(wanted)))).to_pylist())
(root/'results/round1/original_mesh_locations.json').write_text(json.dumps(dict(source_revision='ee4fa27ac534725e24fd1a73c0dcdaccdee75819',requested_hashes=len(wanted),matched_hashes=len({r['sha256'] for r in locations}),locations=locations),indent=2)+'\n')
first=next(r for r in subset['assets'] if r['split']=='train');family='/'.join(first['asset_name'].split('/')[:2]);digest=next(r['id'] for r in mapping if r['name']==family);loc=next(r for r in locations if r['sha256']==digest)
rawurl=loc['fileIdentifier'].replace('https://github.com/','https://raw.githubusercontent.com/').replace('/blob/','/',1);rawurl=urllib.parse.quote(rawurl,safe=':/%')
dest=base/'generation_preflight';dest.mkdir(exist_ok=True)
with urllib.request.urlopen(rawurl,timeout=60) as r:data=r.read(64*1024*1024+1)
if len(data)>64*1024*1024 or hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Original mesh hash/size mismatch')
(dest/'asset.fbx').write_bytes(data)
record=dict(asset_name=first['asset_name'],source_url=loc['fileIdentifier'],sha256=digest,bytes=len(data),source_license=loc['license'],scope='Original TRAIN asset preflight only; never train on test meshes; original files and images remain local')
(root/'results/round1/generation_preflight.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(matched_hashes=len(locations),asset_name=first['asset_name'],bytes=len(data))))
