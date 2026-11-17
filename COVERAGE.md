# Paper-to-implementation coverage

Reviewed 2026-11-17. All original experiment tables remain unreplicated. Latest evidence: [round3](results/round3/REPORT.md). The round2 table below is historical; the following update supersedes its missing-component statements. [Round-2 results](results/round2/REPORT.md) distinguish GT-conditioned reconstruction from strict image-only generation.

| Component | Status | Evidence or gap |
|---|---|---|
| Rig-aware pooling, Eq.2 | Implemented and tested | Actual frozen DINO point features pooled by supplied GT skins; reconstruction encoder only |
| Bone 64³ occupancy | Independent implementation | Closed-cell boundary rule explicit; not author-equivalent voxelizer |
| BFS coordinate/parent tokens | Implemented/tested | Global depth z-y-x; invalid-tree rejection; parent constraints |
| Actual DINO multiview features | Original aligned cohort processed | 25/6/10 assets, four neutral224px views, DINOv2-S384; not original DINO-L/reg1024 textured512px setup |
| Surface/rig Sparse VAE | Small coordinate-aware stochastic models trained | Seven-neighbor sparse convolution, reparameterization/KL, 3 seeds×2 VAEs; feature reconstruction, supplied occupied positions; no TRELLIS pretraining or joint mesh/rig losses |
| Rig-aware mesh decoder + FlexiCubes | Small field decoder + actual upstream FlexiCubes trained/exercised | Resolution16, default blend weights, weak nearest-normal SDF supervision; no original sparse upsampling/deformation/rendering losses |
| AR skeleton decoder | Small model trained/audited | Single-asset memorization passes; held-out geometry errors/EOS cap failures remain |
| Point–bone skinning Eq.7 | Small model trained | GT-bone task for accuracy; generated bones only normalization checked; generated-vertex512-query chunks approximate full point attention |
| Four rectified flows | Reduced image-conditioned pilot trained | Direct16³ occupancy and local8 MLP flows; missing occupancy VAEs, flow transformers and cross-branch coupling |
| Strict single-image generation | Executed, poor quality/frequent failure | 3 seeds×10 tests at CFG5:13 nonempty meshes/30;17 empty. CFG1/3 sensitivities retained; no GT rig/shape/count/voxel inputs at runtime |
| Open-vocabulary joint labels | Missing | No Qwen cleanup or CLIP-aligned joint model |
| Tables1–3: original generation metrics | Not reproduced | No original architecture/scale/baselines, only independent small pipeline |
| Tables4–5: original ablations | Not reproduced | Round1 96/16/32 and round2 25/6/10 subset controls differ from paper |
| Table6: semantic labeling | Not reproduced | Label training/evaluation missing |

21 tests pass. Round2 adds nine formal CPU decoder runs, six final MPS feature-VAEs, three MPS flow/field runs, and 90 strict-generation evaluations across CFG1/3/5. CPU replay verifies all nine selected decoder checkpoints and all six VAEs; source and latent hashes retained. Historical MPS decoder explorations are not pooled into formal CPU results. Raw original assets, views and per-point predictions remain local. An exported mesh or passing gradient check is not evidence of paper-level geometry or rigging quality.


## Round3 current coverage

| Component | Executed evidence | Remaining difference |
|---|---|---|
| Official SLat surface encoder | All100 keys loaded, actual DINO-L/reg1024 `x_prenorm`, full12-block768-width native CPU/MPS | Four upsampled224 neutral views and point-pooling; no full Anymate pre-adaptation |
| Official mesh decoder | All120 keys; actual64→128→256 upsampling; learned SDF/deformation/blend parameters; original FlexiCubes extraction | Geometry-only export; no original color/render/normal losses or CUDA parity |
| Official occupancy VAE | All74+74 keys; full64³↔16³×8; test GT IoU0.984/1.000 | Frozen transfer model; no original24-hour occupancy training |
| Joint training | Three2000-step adapter warm starts, then three250-step full rig+mesh weight updates; surface frozen |16-cube geometry patches every8 steps; short SDF surrogate, not original convergence/full rendering supervision |
| Four image-conditioned flows | Coupled3D occupancy flows, two sparse local flows; standard and foreground-loss diagnostics,3 seeds each | Compact newly trained architecture, not original pretrained flow transformers |
| Strict full-weight image generation | Standard5/30 completed; foreground diagnostic23/30, rig IoU0.0032 | Severe occupancy, localization, EOS and degenerate-bone errors; no semantic correctness claim |
| GT support / noise controls | Non-learned GT shell, GT occupancy perturbations, EOS calibration and coordinate-support oracle | All explicitly separate from single-image generation |
| Validation |28 tests,3 full-weight CPU/MPS rig/mesh probes,12 selected flow probes,3 frozen-decoder CPU replays, exact source hashes | Not exhaustive original CUDA optimization replay or original benchmark reproduction |

All previous failures remain. Raw assets/views, generated vertex/skin arrays and large updated checkpoints stay local. All paper tables, original baselines, labels, textures and retargeting remain unreproduced.
