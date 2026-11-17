# Round3: pretrained TRELLIS and full-weight joint audit

Fixed aligned original cohort:25 train /6 validation /10 test,24/6/10 eligible skeletons. All 3 seeds retained; original benchmark tables remain unmatched.

Actual pretrained encoder/mesh/occupancy VAEs and short full-weight joint stage executed on Mac. GT-conditioned reconstruction/oracle quality is not single-image generation. Small fresh flow architecture, patch SDF loss, short training and missing original data/metrics/labels prevent a complete paper reproduction.

## GT-conditioned reconstruction and support controls

Official pretrained encoder+mesh: 10/10 nonempty, mesh CD×100=0.987. GT-support zero-channel shell (no learned decoder): 10/10 nonempty, CD×100=1.311. GT positions explain a substantial part of low reconstruction distance; neither result is image generation.

Official occupancy VAE GT-mask reconstruction IoU: surface=0.983796, rig=1.000000. Decoder inputs are GT-encoded latents, not generated structure.

| Three-seed GT-context task | Mesh CD×100 | Free J2J×100 | GT-bone element L1×100 | Cap hits /30 |
|---|---:|---:|---:|---:|
| Frozen latent decoder control | fixed official mesh above | 15.174 ± 1.401 | 3.581 ± 0.115 | 1/30 |
| Joint adapters | 1.109 ± 0.001 | 14.239 ± 0.671 | 3.486 ± 0.180 | 15/30 |
| Full-weight short stage | 1.179 ± 0.067 | 14.091 ± 1.056 | 3.301 ± 0.120 | 15/30 |

Mesh CD uses2048 area-sampled mesh points vs8192 source points, unsquared symmetric mean/2; original paper uses a different render/depth sampling protocol. All-case empty-mesh penalty sqrt(3), unit-box diameter. Paired GT-bone skin rows use identical SHA-seeded512 point queries. No generated-topology skin accuracy is claimed.

## Strict single-image generation

These cases read only one image condition and noise at runtime. All empty/budget failures are retained. Main objective is standard MSE; foreground-balanced loss is an explicitly changed diagnostic, not a paper result.

| Path | Nonempty meshes /30 | Surface IoU | Rig IoU | All-case mesh CD×100 | Cap hits / decoded |
|---|---:|---:|---:|---:|---:|
| Adapters / standard MSE | 5/30 | 0.004851 ± 0.004113 | 0.000093 ± 0.000160 | 147.141 ± 32.337 | 3/5 |
| Adapters / balanced rig | 23/30 | 0.006746 ± 0.006470 | 0.003212 ± 0.000304 | 50.438 ± 64.950 | 9/23 |
| Full weights / standard MSE | 5/30 | 0.004851 ± 0.004113 | 0.000093 ± 0.000160 | 147.229 ± 32.163 | 4/5 |
| Full weights / balanced rig | 23/30 | 0.006746 ± 0.006470 | 0.003212 ± 0.000304 | 50.492 ± 64.908 | 6/23 |

Failure categories (zero structure is not counted as an attempted empty mesh):

- Adapters / standard MSE: `{"empty_predicted_rig_occupancy": 10, "empty_predicted_surface_occupancy": 15, "none": 5}`
- Adapters / balanced rig: `{"empty_predicted_surface_occupancy": 7, "none": 23}`
- Full weights / standard MSE: `{"empty_predicted_rig_occupancy": 10, "empty_predicted_surface_occupancy": 15, "none": 5}`
- Full weights / balanced rig: `{"empty_predicted_surface_occupancy": 7, "none": 23}`

Full-weight-stage local flows are re-aligned to updated rig latents; their occupancy weights and scaling stay bitwise unchanged. Thus unchanged occupancy failures cannot be credited as fixes by full mesh/rig finetuning. Nonempty exports are not semantic fidelity or valid animation.

## EOS / localization diagnostics

| Adapter seed | Validation-selected threshold | Main cap hits /10 | Calibrated cap hits /10 | Main J2J×100 | Calibrated J2J×100 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.5 | 0 | 0 | 15.014 | 15.014 |
| 1 | 0.3 | 10 | 10 | 13.843 | 13.843 |
| 2 | 0.4 | 5 | 3 | 13.859 | 14.100 |

Across adapter seeds, teacher matched-joint L2=0.3403; GT-rig-supported candidate control=0.3493. Restricting candidates to GT-rig support is an oracle diagnostic and did not repair the sequence localization. GT quantized-joint to rig-voxel-center distance=0.0072, versus free joints=0.0742. EOS/quantization alone do not explain the remaining coordinate error.

## What remains unmatched

- Native kernels retain original weights/architecture, but no original CUDA rendering parity run exists. Actual features are DINO-L/reg x_prenorm; four native224 neutral views are upsampled518 and point-pooled, unlike full original rendering/preprocessing.
- Adapter stage freezes mesh backbone; subsequent short stage updates full rig+mesh weights but uses only250 steps and16-cube geometry patches every8 steps after a2000-step warm start. Not original convergence or full rendering/normal/color loss.
- Occupancy VAEs are frozen official transfer models; no original24-hour retraining. The four generator networks are compact newly trained conv/attention models, not pretrained TRELLIS flow transformers.
- AR/skin heads remain small width64/two layers; exact coordinate-head factorization and EOS weighting are implementation assumptions. Full-vertex skin inference uses512-query chunks.
- Source cohort is25/6/10 with availability/registration exclusions, not the full benchmark. No labels, textures, retargeting, official baselines or complete original figure/table match.

[Rebuild, source hashes, numerical checks and strict inference boundaries](../../docs/round3_experiments.md). All unfavorable seeds and earlier exploratory paths remain available.
