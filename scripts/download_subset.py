"""Fetch deterministic assets from original splits using data-only torch ZIP views."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import threading
import zlib
import numpy as np
from rigel3d.remote import RangeReader
from rigel3d.torchzip import read_metadata,materialize,TensorRef
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--train',type=int,default=96);ap.add_argument('--val',type=int,default=16);ap.add_argument('--test',type=int,default=32);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
base=root/'data/external';out=base/'subset';out.mkdir(exist_ok=True)
records=[];lock=threading.Lock();done=0;transferred=0
for official in ['train','test']:
    folder=base/official;inv=json.loads((folder/'inventory.json').read_text());items=read_metadata((folder/'data.pkl').read_bytes())
    entries={e['name']:e for e in inv['entries']};prefix=inv['prefix']
    # Pick independent of model results, without filtering by topology or complexity.
    order=sorted(range(len(items)),key=lambda i:hashlib.sha256(('rigel-subset-v1:'+items[i]['name']).encode()).hexdigest())
    selected=[(i,'train' if n<args.train else 'val') for n,i in enumerate(order[:args.train+args.val])] if official=='train' else [(i,'test') for i in order[:args.test]]
    url=f"https://huggingface.co/datasets/yfdeng/Anymate/resolve/{inv['revision']}/{inv['filename']}"
    def fetch(item):
        global done,transferred
        idx,split=item;row=items[idx];asset_hash=hashlib.sha256(row['name'].encode()).hexdigest();dest=out/f'{split}_{asset_hash[:16]}.npz'
        infofile=dest.with_suffix('.json')
        if dest.exists() and infofile.exists():
            rec=json.loads(infofile.read_text());assert rec['asset_name']==row['name']
            assert hashlib.sha256(dest.read_bytes()).hexdigest()==rec['sha256'],'Cached asset checksum mismatch'
            return rec
        rr=RangeReader(url,inv['size']);storages={};checks=[]
        def storage(key):
            if key not in storages:
                name=prefix+'data/'+key;e=entries[name];rr.seek(e['offset'])
                raw=rr.read(min(30+len(name.encode())+512+e['size'],inv['size']-e['offset']))
                if raw[:4]!=b'PK\x03\x04':raise ValueError('Not a local ZIP header')
                namelen,extralen=struct.unpack_from('<HH',raw,26);start=30+namelen+extralen;content=raw[start:start+e['size']]
                if len(content)!=e['size'] or zlib.crc32(content)!=e['crc32']:raise ValueError('Storage CRC mismatch')
                storages[key]=content;checks.append(dict(member=name,size=len(content),crc32=e['crc32'],sha256=hashlib.sha256(content).hexdigest()))
            return storages[key]
        fields=['pc','joints','conns','bones','skins_index','skins_weight']
        data={k:materialize(row[k],storage) for k in fields}
        data.update(name=np.array(row['name']),official_split=np.array(official),source_index=idx)
        np.savez_compressed(dest,**data)
        rec=dict(asset_name=row['name'],split=split,official_split=official,source_index=idx,filename=dest.name,
            sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_file=inv['filename'],source_revision=inv['revision'],
            storage_checks=checks,transferred_bytes=rr.bytes_received)
        infofile.write_text(json.dumps(rec,indent=2)+'\n')
        with lock:
            done+=1;transferred+=rr.bytes_received
            print(json.dumps(dict(official=official,done=done,transfer_bytes=transferred,asset=row['name'])),flush=True)
        return rec
    with ThreadPoolExecutor(max_workers=args.workers) as pool:records.extend(pool.map(fetch,selected))
assert len({r['asset_name'] for r in records})==len(records),'Cross-split duplicate ID'
manifest=dict(revision=inv['revision'],selection='sha256(rigel-subset-v1:asset_name), ascending; train/val from official train shard0, test from official test; no complexity or outcome filtering',
    counts=dict(train=args.train,val=args.val,test=args.test),assets=records)
(root/'results/anymate_subset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(counts=manifest['counts'],transfer_bytes=sum(r['transferred_bytes'] for r in records))),flush=True)
