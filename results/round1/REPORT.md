# Original Anymate subset: all-seed results

Real Anymate subset: 96 train/16 validation/32 test; 83/14/32 eligible single-root skeletons; 2000 steps, three seeds per condition, width64/layers2, 256 train queries and 512 eval queries

Held-out reconstruction proxy; GT rig geometry remains a conditioning input. Rig-context mode also uses GT skin weights to form context. Not image-conditioned generation or original-paper numerical reproduction.

Distances and L1 below are multiplied by 100 for readability. Mean ± sample SD across three seeds. They are not comparable to image-conditioned paper tables.

| Condition | Skin element L1 ×100 | J2J ×100 | J2B ×100 | B2B ×100 | Cap hits /96 |
|---|---:|---:|---:|---:|---:|
| Rig reconstruction | 3.484 ± 0.084 | 13.341 ± 1.167 | 12.361 ± 1.348 | 11.169 ± 1.132 | 96/96 |
| Geometry rig | 3.490 ± 0.070 | 13.575 ± 1.064 | 12.599 ± 1.221 | 11.252 ± 1.101 | 96/96 |
| Balanced EOS | 3.587 ± 0.095 | 15.015 ± 1.281 | 14.090 ± 1.487 | 12.514 ± 1.356 | 2/96 |

Bone-distance control: skin element L1 ×100 = 3.501; train-rig surface retrieval: J2J ×100 = 10.015. These are local controls, not the paper baselines.

The unweighted-stop settings reach the cap frequently. Balanced EOS is an explicitly changed diagnostic loss; its improvement in stopping does not establish improved geometry. No unsuccessful run is replaced by a favorable seed.

Skinning evaluation uses supplied GT bone geometry and point correspondence. The rig reconstruction context also contains GT-skin-derived pooled features. No full image-to-asset generation, learned Sparse VAE, geometry decoder or paper table is reproduced.

Original raw files, pointwise predictions and pretrained weights remain local and ignored by Git. Configs, traces, hashes, exclusions and per-asset scalar metrics are retained.
