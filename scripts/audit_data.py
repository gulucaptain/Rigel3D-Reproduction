"""Independent verification of saved original/prepared assets and held-out IDs."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
root=Path(__file__).resolve().parents[1]
source=json.loads((root/'results/anymate_subset_manifest.json').read_text());prepared=json.loads((root/'results/anymate_preparation.json').read_text())
family=collections.defaultdict(set);checks=[]
for r in source['assets']:
    path=root/'data/external/subset'/r['filename'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
    family['/'.join(r['asset_name'].split('/')[:2])].add(r['split'])
for r in prepared['assets']:
    path=root/'data/external/prepared'/r['filename'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['prepared_sha256']
    d=dict(np.load(path));valid=d['skin_valid'];w=d['weights']
    assert np.isfinite(w).all() and (w>=0).all()
    assert np.max(np.abs(w[valid].sum(-1)-1))<1e-5
    endpoint_error=float(cKDTree(d['joints']).query(d['bones'].reshape(-1,3))[0].max())
    assert endpoint_error<1e-6
    if len(d['tokens']):assert d['tokens'][0,3]==0 and (d['tokens'][1:,3]<np.arange(1,len(d['tokens']))).all()
    checks.append(dict(asset_name=r['asset_name'],split=r['split'],endpoint_to_joint_max_error=endpoint_error,
        valid_weight_points=int(valid.sum()),excluded_weight_points=int((~valid).sum()),skeleton_status=r['skeleton_status'],reason=r['skeleton_exclusion_reason']))
assert all(len(v)==1 for v in family.values()),'Cross-split source-family leak'
report=dict(passed=True,assets=len(checks),asset_ids_disjoint=True,source_families_disjoint=True,
    official_split='train and val exclusively original train0; test exclusively original test',
    exclusions=dict(collections.Counter(r['reason'] for r in checks if r['skeleton_status']!='eligible')),checks=checks)
(root/'results/round1/data_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
