# Pinned upstream FlexiCubes

Source: https://github.com/MaxtirError/FlexiCubes/tree/815e075a2a400d06c48d94c347674344ed6ae5c5

This is the submodule revision referenced by TRELLIS commit 193e0502b5cb3eb5997ec457d5340e95d09b158e. Apache-2.0; copyright NVIDIA, retained in each source file and LICENSE.txt.

Only modification to flexicubes.py: replace `from kaolin.utils.testing import check_tensor` with `from .compat import check_tensor`. The local replacement checks tensor shape. Geometry/topology and differentiable operations are unchanged. tables.py is unchanged. This is actual FlexiCubes, not marching cubes or marching tetrahedra under another name.
