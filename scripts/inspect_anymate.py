"""Inspect public torch ZIP structure without unpickling external objects."""
import argparse
import json
from pathlib import Path
import pickletools
import zipfile
from rigel3d.remote import RangeReader
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','test'],default='train');a=ap.parse_args()
files=json.loads((root/'sources/anymate-tree.json').read_text());name='Anymate_train_0.pt' if a.split=='train' else 'Anymate_test.pt'
size=next(x['size'] for x in files if x['path']==name);revision='096e25372f38db1964c5f5ecaa68c45d78f72ec8'
reader=RangeReader(f'https://huggingface.co/datasets/yfdeng/Anymate/resolve/{revision}/{name}',size)
base=root/'data/external'/a.split;base.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(reader) as z:
    infos=z.infolist();meta=next(i for i in infos if i.filename.endswith('/data.pkl'))
    print(json.dumps(dict(entries=len(infos),metadata_bytes=meta.file_size,metadata_compressed=meta.compress_size,first=[dict(name=i.filename,size=i.file_size) for i in infos[:6]])),flush=True)
    if meta.file_size>64*1024*1024:raise ValueError('Metadata too large for bounded inspection')
    raw=z.read(meta);(base/'data.pkl').write_bytes(raw)
    globals_=sorted({str(arg) for op,arg,_ in pickletools.genops(raw) if op.name=='GLOBAL'})
    (base/'inventory.json').write_text(json.dumps(dict(revision=revision,filename=name,size=size,prefix=meta.filename[:-len('data.pkl')],
        globals=globals_,entries=[dict(name=i.filename,size=i.file_size,offset=i.header_offset,crc32=i.CRC) for i in infos]),indent=2)+'\n')
    print(json.dumps(dict(globals=globals_,transfer_bytes=reader.bytes_received,requests=reader.requests)),flush=True)
