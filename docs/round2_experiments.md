# Round 2: DINO, sparse latents, EOS and image-only generation

This is an independent small-scale implementation. [Results and all seed aggregates](../results/round2/REPORT.md), [machine-readable summary](../results/round2/summary.json), [paper](https://arxiv.org/abs/2605.13129). The local implementation is unpublished; no original paper table is numerically matched.

## What each experiment receives

| Experiment | Inputs at inference | Claim boundary |
|---|---|---|
| Single training-asset memorization | GT geometry/rig context; no joint prefix/count | AR/EOS implementation control, not held-out performance |
| Geometry / DINO / VAE decoder comparison | GT sparse positions and rig context (including GT skin pooling); skinning also uses GT bones | Held-out rig-conditioned reconstruction; free AR means no joint prefix/count, not image-only generation |
| Sparse VAE feature reconstruction | GT active voxel coordinates + multiview DINO features | Coordinate-aware learned stochastic feature compression, not predicted occupancy |
| GT-latent mesh reconstruction | GT sparse positions and encoded GT-rig features | Learned field/FlexiCubes reconstruction diagnostic |
| GT signed-field extraction | GT nearest-triangle field | Resolution/extraction oracle, no learned prediction |
| Strict image-only generation | One image's frozen DINO feature, noise, trained weights and train-only latent statistics | Predicts occupancy, local features, field/mesh, AR skeleton, skin weights; no target geometry, rig, prefix or count at inference |

`generate_asset` in `src/rigel3d/inference.py` has no dataset loader and no target argument. `scripts/generate_image.py` loads only the supplied image, frozen DINO and learned checkpoints/statistics. Evaluation accesses GT targets outside that function. Generated-bone skin weights have no agreed same-index correspondence to GT; only finite values, nonnegativity and row normalization are checked. They are not reported as GT-skin accuracy.

## EOS and geometry diagnosis

The single TRAIN-asset control, seed0/2000 steps, memorizes a 37-joint tree: coordinate and parent token accuracy reach 1, free generation produces 37 joints and EOS at the true end. J2J is about 0.00359, close to the quantization floor. This supports correctness on this asset; it does not prove all implementation choices match the author code or that generalization is solved.

Round 1 teacher-forced coordinate errors were already large, so exposure bias alone does not explain free-generation error. Unweighted EOS was dominated by continue tokens; the explicitly changed balanced per-class stop loss fixed most cap events but did not fix geometry. Round 2 holds balanced EOS, architecture width64/two layers, 48 context tokens, 2000 steps, batch4, 256/512 train/eval points and seeds0/1/2 fixed across the three context conditions. No test-based early stopping or favorable-seed selection. We retain teacher-forced geometry, GT-quantization and GT-count-prefix diagnostics separately from main free-generation metrics.

The same small model times at 0.02694 s/step on Mac CPU versus 0.08267 on MPS in a synthetic shape-matched audit (not a paper speed comparison). The formal paired decoder experiments use CPU throughout. Earlier interrupted MPS runs are explicitly historical (`round2_geometry_pre_sync_optimization`, `round2_geometry_mps_exploration`), never merged into the formal aggregate. DINO, VAEs, flows and field training use MPS. Gradient checks were batched to reduce device synchronization without changing their meaning.

## Actual original assets and DINO

Source Anymate revision and deterministic original subset are unchanged from [round 1](anymate_experiments.md). This follow-up takes the first 32 train /8 validation /12 test candidates from that fixed manifest before model training. Original meshes are resolved using the fixed Anymate family mapping and Objaverse-XL GitHub index, downloaded at fixed source commits, and checked against source SHA-256. Original mesh licenses are not assumed from the dataset-card license. Raw source meshes, rendered views, per-point data and derived qualitative asset views remain local and ignored by Git.

48/52 source files downloaded; 45 imported, 41 passed fixed surface-only registration gates. The resulting cohort is 25/6/10, with 24/6/10 skeleton-eligible assets. Full per-candidate failures remain in `dino_cohort_sources.json`, `mesh_export.json`, `mesh_alignment.json`. ICP uses surface geometry only, not joints or skin weights; acceptance thresholds median<0.012 and p90<0.035 were fixed before training. This test subset is selected for successful source integration and is not the full benchmark.

FBX evaluates animation at the dataset frame, exports evaluated triangles and uses the official Y-up coordinate mapping. The mesh is registered to the previously normalized point cloud. That preparation uses the existing GT-union normalization convention for reconstruction labels; it does not establish the author's undisclosed normalization. Benchmark input images are rendered in that prepared frame. Image-only runtime does not read GT normalization or shape extents.

Blender Workbench renders four neutral-material 224×224 orthographic views, not original textured TRELLIS renders. Visibility uses first-hit rays with 0.02 distance tolerance (registration residuals are nonzero). Bilinear DINO patch projection averages visible views; unseen surface points copy the nearest visible surface feature. Per-asset fractions/hashes are retained. Overall visibility is 60.29–97.22%, mean 85.34%; nearest-feature filling is a stated approximation, not visibility ground truth.

Frozen official `dinov2_vits14`, repository revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, weights SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`, supplies actual 384-channel features. Surface pooling uses all occupied point-sampled 64³ cells. Bone features use Eq.2 GT-skin-weighted point features and bone rasterization. The original TRELLIS config uses DINOv2-L/reg, 1024 channels and 512px images; ours is deliberately smaller. A point-sampled voxelizer can miss triangle-intersected cells.

## Sparse VAE

`src/rigel3d/sparse_vae.py` implements sparse coordinate lookup, seven axial-neighbor submanifold convolution, residual blocks, posterior mean/log-variance, stochastic reparameterization, reconstruction and analytic KL. Missing neighbors contribute zero; occupied coordinates are preserved. A direct dense reference checks sparse-neighbor outputs/gradients. Unlike the original autoencoder, no TRELLIS encoder/mesh-decoder pretrained initialization is used. This is a native sparse VAE component, not a renamed coordinate-blind MLP, and not the original TRELLIS sparse transformer.

Separate surface/rig models: width32, two encoder/decoder blocks, latent8, AdamW3e-4, KL weight1e-4, 4000 steps, seeds0/1/2. Feature channel mean/std are computed only on train voxels; model selection minimizes validation feature MSE. 600-step exploration is retained separately. Downstream decoders use frozen deterministic posterior means, not a joint end-to-end original autoencoder. Test feature MSE is averaged per asset; a train-mean zero-predictor baseline is recorded in `data_audit.json`.

## Four flows and actual FlexiCubes

Four independently parameterized rectified-flow models learn surface occupancy, rig occupancy and their two local 8-channel latents. Image conditions are pooled real DINO patch features from one randomly selected view, with 10% condition dropout. Validation uses fixed noise and view0; inference uses 25 Euler steps and CFG5. Generated occupancy determines positions; a fixed 256-cell safety cap is independent of GT counts. Empty predicted occupancy remains an explicit failure. This reduced implementation directly models 16³ occupancy, and lacks the original occupancy-compression VAEs, sparse flow transformers and cross-branch conditioning. It is a small complete input/output path, not the full original generation architecture.

The learned field decoder queries a fixed 17³ grid and cross-attends to surface and rig latents. Supervision is clipped nearest-triangle distance with nearest-normal sign; signs can be unreliable on open or inconsistently oriented meshes. It lacks original rendering/normal losses and learned FlexiCubes blend/deformation parameters. Generated-vertex skinning uses 512-point chunks, an approximation to full-vertex point self-attention; positional encodings are concatenated with features rather than reproducing the paper’s additive projection formula. Default FlexiCubes weights extract the predicted field; resolution16 is much lower than the paper representation.

Upstream FlexiCubes is fixed at `815e075a2a400d06c48d94c347674344ed6ae5c5`, the submodule in TRELLIS `193e0502b5cb3eb5997ec457d5340e95d09b158e`. Apache-2.0 copyright/license retained. Only Kaolin's shape-check import is replaced with a local shape check; geometry code unchanged. CPU and MPS sphere extraction/backward both pass. Generated meshes are extracted on CPU for portability; network inference remains MPS. No marching-cubes substitute is labeled FlexiCubes.

## Rebuild

After the round-1 data installation described in `anymate_experiments.md`:

```sh
.venv/bin/python scripts/download_dino_cohort.py
/Applications/Blender.app/Contents/MacOS/Blender --background --disable-autoexec --python scripts/export_dino_meshes.py
.venv/bin/python scripts/align_dino_meshes.py
/Applications/Blender.app/Contents/MacOS/Blender --background --disable-autoexec --python scripts/render_dino_cohort.py
.venv/bin/python scripts/extract_dino_cohort.py
/Applications/Blender.app/Contents/MacOS/Blender --background --disable-autoexec --python scripts/prepare_sdf_targets.py
.venv/bin/python scripts/train_sparse_vae.py --steps 4000 --name vae4000
sh scripts/run_round2_decoders.sh
.venv/bin/python scripts/train_generation_pilot.py
.venv/bin/python scripts/audit_round2_data.py
.venv/bin/python scripts/verify_vae_checkpoint.py
.venv/bin/python scripts/audit_evaluations.py --suites round2_geometry round2_dino round2_vae
.venv/bin/python scripts/verify_checkpoint.py --suite round2_geometry
.venv/bin/python scripts/verify_checkpoint.py --suite round2_dino
.venv/bin/python scripts/verify_checkpoint.py --suite round2_vae
.venv/bin/python scripts/evaluate_generation_pilot.py
.venv/bin/python scripts/evaluate_generation_pilot.py --cfg 1 --output-name generation_cfg1
.venv/bin/python scripts/evaluate_generation_pilot.py --cfg 3 --output-name generation_cfg3
.venv/bin/python scripts/audit_generation_shift.py
.venv/bin/python scripts/audit_image_condition.py
.venv/bin/python scripts/report_round2.py
.venv/bin/python -m pytest -q
```

Training refuses existing result directories. Preserve/rename old experiment directories and ignored latent caches before retraining; don't overwrite formal results silently. Download/render/extraction stages reuse existing files; verify the recorded hashes before relying on a cache. `scripts/audit_ar_overfit.py` and `scripts/audit_flexicubes.py` regenerate their diagnostic reports and must be run in a copied experiment tree to preserve history.

For a new image after training:

```sh
.venv/bin/python scripts/generate_image.py --image /absolute/path/input.png --output /absolute/path/new-output --seed 0 --noise-seed 123
```

This exports `mesh.obj`, `skeleton.json`, `asset.npz` (including generated skinning) and `status.json`. A file export is not evidence of visual/rigging quality. Read the failure flags and all-case held-out results first. Joint labels, textures, retargeting and original benchmark baselines remain outstanding.

## Current evidence and unresolved causes

Real DINO did not improve J2J over the geometry proxy in this small paired pilot; VAE feature compression improves over a constant predictor but is not joint/mesh accuracy. CFG1/3/5 gives16/15/13 nonempty meshes out of30, with similar skeleton error. Thus guidance changes alone do not repair generation. CFG5 coarse occupancy IoU averages0.2349(surface)/0.0533(rig); normalized generated latent RMS1.658/1.441 versus training1 supports distribution-shift and occupancy-error diagnoses. It does not identify a unique cause: missing pretrained mesh priors, limited data, coarse occupancy, sparse-context truncation and separate training remain confounded. Replacing components with original pretrained counterparts and restoring joint autoencoder supervision is the next evidence-supported direction.
