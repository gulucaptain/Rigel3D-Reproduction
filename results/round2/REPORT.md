# Round 2: real DINO, Sparse VAE and strict image-only pilot

Fixed source cohort:25 train/6 val/10 test,24/6/10 eligible skeletons; three seeds all reported. Decoder reconstruction uses GT rig geometry/weights; strict generation only uses one image plus noise.

Independent small pipeline verification, not original architecture/scale, paper baselines or numerical reproduction. All original paper tables remain unmatched.

Mean ± sample SD across seeds. All distances below ×100. No direct comparison with original paper tables.

| GT-context reconstruction | Skin L1 element | J2J | J2B | B2B | Cap hits /30 |
|---|---:|---:|---:|---:|---:|
| Geometry proxy | 3.236 ± 0.235 | 13.722 ± 1.315 | 12.767 ± 1.501 | 12.103 ± 1.664 | 9/30 |
| Real DINO | 3.730 ± 0.177 | 14.651 ± 1.042 | 13.781 ± 1.159 | 12.587 ± 1.056 | 6/30 |
| Sparse VAE | 3.450 ± 0.330 | 14.223 ± 1.444 | 13.183 ± 1.513 | 12.237 ± 1.509 | 6/30 |

## Sparse VAE component

Normalized real-DINO feature test MSE: surface 0.7510 ± 0.0067; rig 0.7024 ± 0.0095. Ground-truth active coordinates are retained. This is feature reconstruction, not learned occupancy or mesh generation.

## Strict image-only generation

Nonempty extracted meshes: 13/30; reported failure flags: 17/30. Every failure remains in per-asset reports.

| Path | Mesh Chamfer ×100 | J2J ×100 |
|---|---:|---:|
| GT-latent reconstruction | 14.906 ± 9.801 | not this diagnostic |
| Image-only generation | 105.535 ± 39.816 | 14.961 ± 1.377 |

Empty mesh/skeleton cases receive sqrt(3), the unit-box diameter, in the all-case diagnostic aggregate; nonempty-only distances and denominators are also retained. Chamfer uses 2048 area-weighted sampled mesh points, 8192 target points, unsquared symmetric mean/2. Skinning on generated topology is checked for normalization only: no same-index GT-skin accuracy claim.

## Guidance and distribution diagnostics

The CFG5 setting remains the main result. The following are matched-noise test sensitivity audits, not a test-selected replacement.

| CFG | Nonempty meshes /30 | 96-joint caps /30 | J2J ×100 |
|---|---:|---:|---:|
| 1 | 16/30 | 9/30 | 14.910 |
| 3 | 15/30 | 10/30 | 15.058 |
| 5 | 13/30 | 10/30 | 14.961 |

CFG5 mean coarse occupancy IoU: surface 0.2349, rig 0.0533. Generated normalized latent RMS: surface 1.658, rig 1.441, versus train RMS=1 by normalization. These show occupancy error and distribution shift; guidance alone does not resolve failures.

## Known gaps

- Random initialization; no pretrained TRELLIS surface encoder or mesh decoder, unlike the original method.
- DINOv2-S 384 channels, four neutral 224px views; original TRELLIS configuration uses DINOv2-L/reg 1024 channels and 512px images.
- Seven-tap axial sparse VAE, width32/8 latent channels; separately trained feature loss+KL, not original joint autoencoder losses or pretrained sparse transformer.
- Direct 16³ occupancy MLP flows lack original occupancy VAEs, sparse flow transformers and cross-branch coupling.
- Independent cross-attention field decoder predicts a 17³ vertex grid; fixed upstream FlexiCubes uses default blend weights, no learned vertex deformations.
- Oriented-nearest-triangle SDF is weak supervision on open/inconsistently oriented meshes; original differentiable rendering losses absent.
- Decoder models are small width64/two-layer, frozen VAE means and GT-rig reconstruction context. The 25/6/10 aligned-asset cohort is not the complete benchmark.
- Semantic joint labels, textures, retargeting and original paper baselines/ablation tables remain unimplemented/unmatched.

See docs/round2_experiments.md for source, exclusions, commands and inference boundaries.
