import hashlib
import json
from pathlib import Path
import numpy as np
from rigel3d.data import prepare_asset
root=Path(__file__).resolve().parents[1];m=json.loads((root/'results/anymate_subset_manifest.json').read_text())
out=root/'data/external/prepared';out.mkdir(exist_ok=True);records=[]
for r in m['assets']:
    path=root/'data/external/subset'/r['filename'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
    data,report=prepare_asset(dict(np.load(path)));np.savez_compressed(out/r['filename'],**data)
    records.append(dict(asset_name=r['asset_name'],split=r['split'],filename=r['filename'],prepared_sha256=hashlib.sha256((out/r['filename']).read_bytes()).hexdigest(),**report))
report=dict(source_manifest_sha256=hashlib.sha256((root/'results/anymate_subset_manifest.json').read_bytes()).hexdigest(),
    scope='Original train0/test assets; explicit target normalization and GT-rig reconstruction context, not image-conditioned generation',assets=records)
(root/'results/anymate_preparation.json').write_text(json.dumps(report,indent=2)+'\n')
for split in ['train','val','test']:
    rows=[r for r in records if r['split']==split]
    print(json.dumps(dict(split=split,assets=len(rows),skeleton_eligible=sum(r['skeleton_status']=='eligible' for r in rows),
        skin_valid_points=sum(r['skin_valid_points'] for r in rows))),flush=True)
