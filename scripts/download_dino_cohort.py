"""Resolve real original assets for a fixed candidate cohort before any fitting."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
root=Path(__file__).resolve().parents[1];base=root/'data/external/dino_cohort';base.mkdir(exist_ok=True)
source=json.loads((root/'results/anymate_subset_manifest.json').read_text());mapping={r['name']:r for r in json.loads((root/'data/external/Anymate_id.json').read_text())}
candidates=[]
for split,count in [('train',32),('val',8),('test',12)]:
    for r in [r for r in source['assets'] if r['split']==split][:count]:
        fam='/'.join(r['asset_name'].split('/')[:2]);candidates.append(dict(**r,raw_sha256=mapping[fam]['id']))
wanted={r['raw_sha256'] for r in candidates};locations=[]
for batch in pq.ParquetFile(root/'data/external/objaverse-github.parquet').iter_batches(batch_size=65536,columns=['fileIdentifier','sha256','license','fileType']):
    t=pa.Table.from_batches([batch]);locations.extend(t.filter(pc.is_in(t['sha256'],value_set=pa.array(sorted(wanted)))).to_pylist())
bysha={r['sha256']:r for r in locations}
def fetch(r):
    out=dict(asset_name=r['asset_name'],split=r['split'],filename=r['filename'],raw_sha256=r['raw_sha256'])
    if r['raw_sha256'] not in bysha:return dict(**out,status='source_unresolved')
    loc=bysha[r['raw_sha256']];folder=base/Path(r['filename']).stem;folder.mkdir(exist_ok=True)
    path=folder/('original.'+loc['fileType']);out.update(source_url=loc['fileIdentifier'],raw_path=str(path.relative_to(root)),source_license=loc['license'])
    try:
        if not path.exists():
            url=loc['fileIdentifier'].replace('https://github.com/','https://raw.githubusercontent.com/').replace('/blob/','/',1);url=urllib.parse.quote(url,safe=':/%')
            with urllib.request.urlopen(url,timeout=60) as response:data=response.read(64*1024*1024+1)
            if len(data)>64*1024*1024:raise ValueError('asset exceeds 64MiB bound')
            if hashlib.sha256(data).hexdigest()!=r['raw_sha256']:raise ValueError('source hash mismatch')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['raw_sha256']:raise ValueError('cached source hash mismatch')
        out.update(status='downloaded',bytes=path.stat().st_size)
    except Exception as e:out.update(status='download_failed',reason=type(e).__name__+': '+str(e)[:120])
    return out
with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(fetch,candidates))
folder=root/'results/round2';folder.mkdir(exist_ok=True)
manifest=dict(selection='First 32 train /8 validation /12 test records from existing fixed official-split manifest; record all source failures; no metric-based selection',assets=records)
(folder/'dino_cohort_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({split:dict(candidate=sum(r['split']==split for r in records),downloaded=sum(r['split']==split and r['status']=='downloaded' for r in records)) for split in ['train','val','test']}))
