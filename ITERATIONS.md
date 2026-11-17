# Implementation history

## 2026-11-16 · Initial representation checks

Eight tests and untrained attention CPU/MPS checks. No original data training in this initial evidence; files retained unchanged.

## 2026-11-16 · Original Anymate subset training

- Fixed official source revision, bounded storage extraction and data-only metadata reconstruction. Found 8192 points, parent vectors and sparse skinning fields; corrected the adapter instead of following inconsistent README dimensions.
- Deterministic 96 train /16 validation /32 test; no source-family collision. Retained all selected assets, with skeleton-invalid forests/cycles logged (83/14/32 eligible), zero-weight skin rows explicitly masked.
- Added trainable AR decoder and skinning decoder with PE, masks, shifted teacher forcing, constrained parent topology and EOS. Shared geometry feature projections are substitutes, not pretrained Sparse VAE SLats.
- Six original decoder pilot runs (two contexts×three seeds), 2000 steps, MPS. Validation selects checkpoints; all test seeds reported. Unweighted EOS failed in 96/96 tests per setting; preserved failures.
- Added explicit balanced-EOS control, three more seeds. It improves stopping to 2/96 cap hits but worsens J2J; no altered run replaces original results.
- Preserved first metric definitions in evaluation.json; audit computes separate RigNet-compatible sampled metrics in evaluation_metrics_v2.json and denser-sampling sensitivity. Both element/vector L1 and separate vector L2/MSE recorded.
- Data audit and all nine selected checkpoints replay on CPU; 16 tests pass. Original dataset geometry remains local; public overview only receives reviewed scalar evidence and aggregate figure.
- Located raw source mapping for 112 training/validation families; 101 resolved in fixed Objaverse-XL GitHub index. Verified first training FBX hash; four-view Blender and official frozen DINOv2 extraction complete.
- Retained initial failed zero-visibility projection; corrected raw-versus-evaluated FBX mesh pose mismatch and added nonempty-coverage check. Final sparse feature preparation: 1253/1871 observed vertices, 1028 active surface voxels, 384 channels. Not used to relabel the nine geometry-feature experiments.
- Full Sparse VAE, FlexiCubes, four flow models and label model remain outstanding; no complete original benchmark reproduction.

## 2026-11-17 · Actual DINO, sparse VAE and image-only generation pilot

- Diagnosed AR/EOS through a single TRAIN-asset2000-step control:37 predicted joints, exact coordinate/parent token accuracy and correct EOS. Near-quantization-floor geometry establishes capacity on one memorized asset, not held-out correctness.
- Fixed candidate cohort32/8/12 before training.48 meshes downloaded/hash-checked,45 imported,41 pass surface-only registration gates:25/6/10, skeleton-eligible24/6/10. All failures retained. Actual DINOv2-S features replace geometric proxies in separate new runs; four-view visibility/filling explicitly audited.
- Implemented native coordinate-aware seven-neighbor sparse VAEs, reparameterization/KL and frozen posterior means. Six4000-step surface/rig MPS runs, seeds0/1/2, with CPU mean/loss/hash replay. Earlier600-step models retained as exploration.
- Nine formal2000-step decoder runs compare geometry, DINO and sparse latents on the same cohort, balanced EOS in all conditions. CPU is faster in timing audit, so formal runs all use CPU; interrupted MPS runs remain historical, not mixed into statistics. Aggregate gradient finite checks reduce redundant synchronization only.
- Real DINO/VAEs do not solve held-out joint geometry: J2J×100=13.722±1.315,14.651±1.042,14.223±1.444. Cap hits9/30,6/30,6/30. All seeds and per-asset failures retained; no favorable feature condition replaces the original results.
- Vendored exact TRELLIS-referenced FlexiCubes revision under Apache-2.0, replacing only Kaolin's shape checker. CPU and native MPS sphere extraction/backward pass. Small cross-attention field decoder and four image-conditioned MLP flows trained for three seeds; architecture/pretraining/loss deviations explicit.
- Strict image-only runtime accepts image feature/noise/checkpoints only, predicts both occupied positions and local latents, then mesh, free AR skeleton and generated-bone skins. CFG5 yields13/30 nonempty meshes,17 empty; GT-latent reconstruction29/30 nonempty is separately labeled. Independent PNG CLI exercised and correctly reports its empty-mesh/cap failure.
- CFG1 and3 matched-noise audits yield16/30 and15/30 nonempty meshes, with similar joint error: guidance alone does not fix generation. Main CFG5 remains unchanged. Low coarse rig occupancy IoU and shifted latent RMS are diagnostics, not proven unique causes.
-21 tests pass; all9 decoder and6 VAE checkpoints independently replay. Original benchmark numerical reproduction, pretrained TRELLIS initialization, occupancy VAEs/flow transformers, original mesh losses, labels/textures/retargeting remain outstanding.


## 2026-11-17 · Official TRELLIS integration and full-weight joint follow-up

- Fixed official source/HF revisions; downloaded/hash-verified actual surface/mesh/occupancy weights. Strict100/120/74/74 key loads. Native sparse window attention/convolution/subdivision have CPU reference tests; FP32 actual-data compute follows measured mixed-fp16 drift, with CPU/MPS encoder probe error4.05e-5.
- Corrected feature interface: official DINO-L/reg1024 x_prenorm, not DINO-S384 normalized tokens. Source224 neutral views remain an explicit limitation; actual flow image conditions preserve pooled8×8 spatial tokens.
- Fixed source cohort25/6/10; no result-driven asset replacement. Official GT occupancy transfer reconstructs test surface/rig IoU0.983796/1.0. Official GT mesh gives10/10 nonempty, CD ×100 =0.987; nonlearned GT-support shell1.311 exposes how much geometry is supplied by GT coordinates.
- First dual-head/constant-LR joint and flow runs remain exploratory. Rechecking paper appendix identified frozen surface latents during joint training; formal adapter stage freezes surface parameters and uses scaled warmup/cosine. Three seeds×2000 steps,16-cube mesh loss every8 steps. Actual gradients through rig encoder head and mesh CA independently checked.
- Measured full-backbone Mac feasibility, then added250-step full rig+mesh updates for all3 seeds after adapter warm starts, surface still frozen. First/last full transformer blocks demonstrably change; all3 large local checkpoints hash/replay, max CPU/MPS mesh-field differences below5.5e-5. This is short patch-SDF training, not full original rendering-loss convergence.
- Standard and separately changed foreground-balanced occupancy losses each train3 flow seeds. Local flows realigned after full encoder updates; occupancy weights/scaling verified bitwise unchanged. No top256 position truncation; empty/budget failures explicit.
- Full-weight standard generation5/30 completed:15 empty-surface,10 empty-rig failures. Balanced diagnostic23/30 completed,7 empty-surface failures; rig IoU0.003212 remains very low. All masks and failed cases retained; none replaced by GT positions or favorable seeds.
- Full joint GT-context J2J ×100 =14.091±1.056 and15/30 cap hits: no robust localization/EOS fix. Validation threshold calibration and GT-rig support candidate controls remain secondary and do not solve geometry.
- Paired skin audit corrects query-set comparability: original joint fixed37 sample diagnostics preserved, comparison tables use the same SHA-seeded512 points as frozen control. Full-weight element L1 ×100 =3.301±0.120, not generated-bone skin accuracy.
-28 tests pass; frozen surface bitwise verified,3 full-backbone numerical probes and12 selected flow probes pass. Standalone single-image full-weight diagnostic exports actual mesh/skeleton/per-vertex skins, with low rig consistency and3 zero-length bones recorded. Pipeline execution is not fidelity.
