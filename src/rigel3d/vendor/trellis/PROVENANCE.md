# Microsoft TRELLIS reference components

Official source: https://github.com/microsoft/TRELLIS/tree/193e0502b5cb3eb5997ec457d5340e95d09b158e . MIT license retained in LICENSE.

- `norm.py`, `spatial.py`, `utils_cube.py`: source snapshots unchanged.
- `occupancy.py`: official `trellis/models/sparse_structure_vae.py`; only three relative import paths point to this local module.
- `utils.py`: dense Conv3d/Linear dtype conversion and zero-module compatibility; eliminates sparse CUDA imports. All actual checkpoint keys load strictly.
- `rigel3d/trellis_native.py`: independent batch-one implementation of the same SLat transformer architecture/parameter names and 64→128→256 sparse subdivision. Native PyTorch window SDPA and gather-based submanifold convolution replace flash_attn/spconv. CPU oracle tests verify convolution outputs/gradients, shifted window attention, APE and subdivision. No original CUDA parity run is available on this Mac.

Weights are fixed to microsoft/TRELLIS-image-large HF revision25e0d31ffbebe4b5a97464dd851910efc3002d96, checked against LFS SHA-256, and remain outside Git. Checkpoints are not reinitialized or partially loaded: all100 encoder,120 mesh,74 occupancy encoder and74 occupancy decoder keys match. Actual data experiments use FP32 computation for numerical consistency; fixed fp16 checkpoint weights are losslessly expanded. Mesh extraction retains original SDF bias, deformation and21 blending channels; color attributes are omitted, not geometry.
