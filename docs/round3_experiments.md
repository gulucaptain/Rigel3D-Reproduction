# Round3: official TRELLIS priors, occupancy VAE and joint adaptation

This round uses actual official pretrained weights and retains all round1/2 failures. [Results](../results/round3/REPORT.md) and [summary](../results/round3/summary.json) distinguish GT-conditioned reconstruction from single-image generation. The implementation remains local/unpublished; all original paper tables remain unmatched.

## Source-checked changes

The [Rigel3D appendixB](https://arxiv.org/html/2605.13129v1#A2) describes an Anymate adaptation stage for the TRELLIS surface encoder/mesh decoder, followed by joint rig/mesh/skin training with the surface encoder frozen and its latents pre-extracted. Its latent generators start from pretrained TRELLIS flow weights. The full training scale,5000-step warmup and full rendering losses are not reproduced on this Mac.

Official TRELLIS source is fixed at `193e0502b5cb3eb5997ec457d5340e95d09b158e`; official HF checkpoint revision is `25e0d31ffbebe4b5a97464dd851910efc3002d96`. Four checkpoint/config pairs download622MB, verify LFS SHA-256, and remain ignored by Git. No credential is used for these public downloads. Source hashes and model inventory are in `results/round3/pretrained_sources.json`.

| Official checkpoint | Loaded keys | Parameters | Role |
|---|---:|---:|---|
| SLat encoder |100|85,817,104|1024-channel DINO feature →8-channel surface posterior |
| SLat mesh decoder |120|90,926,693|64→128→256 sparse subdivision,101 output channels |
| Occupancy encoder |74|59,298,192|64³ binary occupancy →16³×8 posterior |
| Occupancy decoder |74|73,671,201|16³×8 posterior →64³ occupancy logits |

Every key is loaded strictly; no missing layers are silently randomized. The mesh101 channels contain8 SDF,24 deformation,21 blending and48 color attributes. Extraction uses original SDF bias−1/256, deformation and21 blending channels with actual upstream FlexiCubes. Color attributes are omitted from output, not substituted into geometry.

## Native Mac port and numerical checks

Original sparse CUDA dependencies are replaced by native PyTorch kernels, keeping pretrained architecture, parameter names and shapes. `trellis_native.py` implements batch-one shifted-window SDPA, integer-coordinate gather submanifold convolution, spatial group normalization and exact8-child subdivision order. Window shifts0/4 alternate across12 blocks; width768 and12 heads are retained. Tests compare convolution outputs/gradients against dense3D convolution, window attention against direct per-window attention, position encoding and subdivision. The full original CUDA backend is unavailable; passing these tests is not full CUDA parity.

Initial mixed-fp16 encoder probes differed by0.0179 between CPU/MPS. FP32 expanded checkpoint values (no reinitialization) reduce the32-token probe max absolute difference to about4.05e-5. Formal actual-data encoding and mesh decoding therefore use FP32. `pretrained_kernel_audit.json` retains the mixed-precision probe; `precision_audit.json` records the choice. Native occupancy VAEs use the official dense code, with only import/utility compatibility changes. The MIT license and provenance are retained.

## Actual DINO-L input and data scope

The same fixed source cohort25 train/6 validation/10 test is retained; skeleton eligibility24/6/10. No new favorable test assets are substituted. Source meshes, original point/rig data, images, dense occupancy caches and generated per-vertex arrays remain local.

The official SLat encoder expects **1024-channel DINOv2-L/reg `x_prenorm`**, not round2's384-channel DINO-S normalized patch output. Real frozen `dinov2_vitl14_reg` is now used with the fixed DINO repository revision and official518 input size. CLS/register tokens are removed before geometric feature projection. All-view mean pooling is used for surface/GT-skin-weighted bone features; round2's visibility-gated feature filling is not mixed into these inputs.

Remaining preprocessing differences: the existing four neutral-material224px Workbench images are upsampled to518, rather than newly textured native high-resolution multiview renders. Features are projected at sampled surface points then voxel-averaged, rather than directly at all triangle-intersected voxel centers. Single-image conditioning spatially pools the37×37 layer-normalized patch map into8×8 tokens; the non-divisible average-pooling operation runs on CPU because MPS does not implement it. DINO itself runs on MPS.

The reconstruction data retains the prior GT-union normalization convention; this is not verified author normalization. Single-image runtime reads no GT normalization or shape bounds. Test input renders are prepared in the benchmark canonical frame, not evidence that arbitrary user photos are normalized correctly.

## Three levels of occupancy evidence

1. **GT occupancy VAE reconstruction:** feed supplied64³ surface/rig occupancy through the official encoder and decoder. Its strong reconstruction is not generated structure and has no claim to image inference.
2. **Generated occupancy:** image/noise → coupled occupancy flows →16³×8 latent grids → official decoder → predicted64³ occupied cells. Compare masks to targets only after the generator returns.
3. **Decoded AR skeleton consistency:** rasterize the free AR skeleton and compare it with the supplied/generated rig-latent occupied cells. A good occupancy mask does not prove the AR decoder follows it.

The original pretrained occupancy VAE is frozen in this pilot; its transfer reconstruction is evaluated on all41 assets. The paper's separate24-hour occupancy-VAE training is not reproduced. The observed transfer reconstruction does not establish generated occupancy quality.

Runtime uses all predicted positions. The old round2 top256-cell truncation is removed. If surface occupancy exceeds16000 cells or rig occupancy exceeds12000, the run returns an explicit budget failure with complete predicted masks; it never substitutes GT positions/counts or silently clips to a favorable subset. Empty predicted structure also remains a failure.

## Joint training scope and historical separation

`results/round3/joint/` stores the first exploratory run that adapted both posterior heads. After rechecking appendixB, the formal run in `joint_frozen_surface/` freezes the surface head as well as its cached original transformer trunk. Formal surface parameters are checked bitwise against official checkpoint values. `flows/` and the single `generation_preflight/` correspond to that earlier exploration, not formal results.

The first formal adapter-stage seeds0/1/2 update the rig posterior head,12 zero-initialized low-width rig cross-attention adapters in the original mesh blocks, the causal skeleton decoder and skinning decoder. The original large mesh transformer/subdivision weights stay frozen. Geometry-loss backward is independently checked: the surface head remains without gradients, rig-CA output receives gradients immediately, and rig-head gradients from geometry appear after the zero-initialized adapter first updates. This is a joint differentiable adapter/head training graph, not separate models run in one script.

Each seed runs2000 steps, batch1,128 training skin queries, width64/two-layer AR/skin heads; geometry updates use16 nearby surface cubes every8 steps (250 patch updates). Cosine LR1e-4, scaled200-step warmup, skeleton/skin loss weights1, KL1e-6 and posterior-anchor weight0.01 are explicit. Cached normalized768-dimensional encoder features are stored in FP16, then used with FP32 heads. Surface latents are therefore unchanged up to documented cache rounding; rig features can adapt.

The geometry objective is nearest oriented GT point distance plus sparse-corner consistency on predicted SDF/deformation. It is **not** the original full-mesh differentiable rendering/normal loss. Patch boundaries and sparse group-normalization statistics differ from full-asset training. Original mesh blending weights are predicted and used at inference; this patch SDF loss does not separately supervise their optimality. No independent Anymate full-encoder/mesh adaptation stage is reproduced.

The `round3_frozen/` decoder-only control uses frozen official posterior means with the same batch1/128-query/2000-step small decoder protocol. It is a local control, not a paper baseline. Validation selects every checkpoint; all three seeds are retained.

## Short full-weight joint stage

A measured three-step feasibility probe updates the actual complete rig encoder and mesh decoder on16 surface input cubes. It uses roughly4.2GB driver memory and about0.9s for the warmed third step; this is a patch workload, not full-asset backpropagation. Thus freezing the mesh backbone is not presented as a hard Mac requirement.

After the2000-step frozen-surface adapter stage, `full_joint/` performs250 additional steps per seed, updating the full85.8M rig transformer and90.9M mesh decoder, rig CA, AR and skin modules. Surface latents remain the original frozen pretrained means. The rig encoder is evaluated on actual1024-channel features, without the old cached-trunk shortcut. The short stage uses AdamW1e-4, weight decay0,25 warmup/cosine, the same16-cube geometry patch every8 steps and the same stated surrogate objective. This verifies full-weight joint updating, not original full-mesh rendering-loss training or convergence.

Large FP32 checkpoints stay in ignored local `data/external/full_joint/`; public-safe manifests store paths, sizes and SHA-256. Verification checks both first/last rig and mesh transformer blocks actually changed, the surface head stayed bitwise frozen, and CPU/MPS rig/mesh field outputs agree. The geometry loss still uses patch SDF/deformation, so this is not the original rendering/normal/color supervision. Standard and foreground-balanced local flows are re-aligned to the updated rig latent space for1000 steps; occupancy weights and train-only scaling remain bitwise unchanged. No GT coordinates are introduced into generation by that re-alignment.

## Flow generation and sparse-foreground diagnostic

Four flow networks model two occupied-structure latents and two local feature sets. The two occupancy branches use compact coupled3D convolutional networks with spatial image-token cross-attention; both see the current noisy surface/rig state. Local flows use image tokens, coordinates and both occupied-structure latent grids. Correct x-y-z versus volume z-y-x interpolation is tested. Training targets come from the frozen official occupancy encoder and validation-selected joint posterior means. Statistics are computed only on train assets.

Standard generator `flows_frozen_surface/`:4000 steps per seed, AdamW1e-4, standard velocity MSE,10% image-condition dropout. Sampling uses25 Euler steps/CFG5. These small randomly initialized flow networks are **not** the original pretrained TRELLIS flow transformers; no model equivalence or original generation quality is claimed.

A post-hoc diagnostic prompted by the exploratory empty-rig failure uses `flows_balanced_rig/`: same seeds, initialization, architecture, budget and training assets, with the rig occupancy velocity loss equally weighted over foreground-support and background16³ sites. Support masks supervise training loss only, not network inputs or inference. This changed loss is not the original paper objective. It is reported separately and is not selected to replace the standard main result based on test quality.

## EOS and coordinate audits

Main free generation starts from BOS, with no GT count/prefix, and uses EOS probability threshold0.5/max96 joints. Balanced EOS training remains an explicit prior diagnostic assumption. Validation-only threshold calibration uses6 validation assets and teacher-forced balanced EOS/continue accuracy on a fixed0.1…0.9 grid; calibrated free outputs are secondary, not replacements for main results. Calibration can worsen test cap events and does not establish improved geometry.

Coordinate support audits compare teacher-forced independent marginal coordinate argmax with an oracle support control restricted to provided GT rig voxels. The latter uses GT geometry and is never image-only generation. Distances from free joints to rig voxel centers and GT-joint reference distances help separate quantization/preprocessing from learned localization errors. Parallel x/y/z categorical heads are an implementation assumption; the paper does not uniquely specify within-joint factorization details.

## GT-support and occupancy-noise controls

A zero-channel256³ shell extracted from the same supplied GT surface occupancy, with no DINO or learned decoder, tests how much mesh Chamfer is explained by GT positions alone. It can make double surfaces and is not a valid generative baseline. Its low distance is a reason not to treat GT-conditioned reconstruction as proof of learned image-to-shape quality.

An occupancy perturbation audit uses the first3 validation assets, three noise seeds, and noise0.1/0.5 times training-channel standard deviation around the GT posterior. It tests only local noise robustness, not arbitrary bias directions or actual generation. Its results cannot turn a GT-conditioned reconstruction into a generated example.

## Inference boundary

`generate_image_only` receives one image-token tensor, trained models, train-only statistics and fixed sampling/budget settings. It has no target/dataset loader. Occupancy, active64³ coordinates and local latents are predicted before the original mesh decoder and free skeleton decoder run. Per-vertex skinning uses generated bones; normalization/degenerate bones are reported, with no same-index GT-skin accuracy claim.512-query chunks approximate full-vertex point self-attention.

`evaluate_joint_trellis.py` is GT-conditioned reconstruction; `evaluate_trellis_generation.py` is single-image generation with target access only in the scoring code after generation. Public plots contain scalar aggregates only. Raw-derived qualitative panels and generated rig arrays stay local.

## Rebuild and reuse

Use the existing environment and original-data preparation in [round2](round2_experiments.md); install `requirements-lock.txt`. Downloads and feature caches may be reused after verifying source hashes. Formal training refuses existing output directories. Preserve old results before retraining rather than overwriting them.

```sh
.venv/bin/python scripts/download_trellis_pretrained.py
.venv/bin/python scripts/extract_trellis_dino.py
.venv/bin/python scripts/cache_trellis_image_tokens.py
.venv/bin/python scripts/audit_trellis_pretrained.py
.venv/bin/python scripts/audit_trellis_precision.py
.venv/bin/python scripts/cache_trellis_latents.py
.venv/bin/python scripts/audit_occupancy_pretrained.py
.venv/bin/python scripts/evaluate_trellis_reconstruction.py
.venv/bin/python scripts/train_subset.py --context trellis --cohort --balanced-stop --device cpu --batch 1 --points 128 --name round3_frozen --steps 2000 --seeds 0 1 2
.venv/bin/python scripts/train_joint_trellis.py
.venv/bin/python scripts/cache_joint_latents.py
.venv/bin/python scripts/train_trellis_flows.py
.venv/bin/python scripts/train_trellis_flows.py --name flows_balanced_rig --balanced-rig-occupancy
.venv/bin/python scripts/evaluate_joint_trellis.py
.venv/bin/python scripts/audit_joint_eos.py
.venv/bin/python scripts/audit_trellis_coordinates.py
.venv/bin/python scripts/evaluate_trellis_generation.py
.venv/bin/python scripts/evaluate_trellis_generation.py --flow-run flows_balanced_rig --name generation_balanced_rig
.venv/bin/python scripts/verify_round3_models.py
.venv/bin/python scripts/train_full_joint.py
.venv/bin/python scripts/cache_full_joint_latents.py
.venv/bin/python scripts/adapt_full_joint_local_flows.py
.venv/bin/python scripts/evaluate_joint_trellis.py --joint-run full_joint --full-joint
.venv/bin/python scripts/evaluate_trellis_generation.py --joint-run full_joint --flow-run flows_full_joint --full-joint --name generation_full_joint
.venv/bin/python scripts/evaluate_trellis_generation.py --joint-run full_joint --flow-run flows_full_joint_balanced --full-joint --name generation_full_joint_balanced
.venv/bin/python scripts/verify_full_joint.py
.venv/bin/python scripts/verify_trellis_flows.py
.venv/bin/python scripts/audit_round3_skin.py
.venv/bin/python scripts/audit_gt_voxel_shell.py
.venv/bin/python scripts/audit_occupancy_robustness.py
.venv/bin/python scripts/report_round3.py
.venv/bin/python -m pytest -q
```

A standalone new-image entry reads only the supplied image and model files:

```sh
.venv/bin/python scripts/generate_trellis_image.py --image /absolute/path/input.png --output /absolute/path/new-output --seed 0 --noise-seed 123 --full-joint --joint-run full_joint --flow-run flows_full_joint
```

`status.json` is authoritative for empty/budget/rig failures. OBJ/NPZ export is not evidence of semantic fidelity, valid animation quality, or original-paper reproduction. Labels, textures, retargeting, original rendering metrics/baselines and original-scale training remain incomplete.

Training optimizer differences are explicit: adapter/control and flow-stage code used AdamW default weight decay0.01; the full-weight stage/local re-alignment set weight decay0, matching the official TRELLIS mesh-training config. The paired skin comparison uses identical SHA-seeded512 query points; earlier joint-evaluation fixed-seed37 skin diagnostics remain unchanged and are not silently compared as paired values.
