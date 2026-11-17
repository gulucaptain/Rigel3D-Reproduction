"""Shape-only compatibility for Kaolin check_tensor; no geometry algorithm changes."""
def check_tensor(tensor, shape, throw=True):
    ok=tensor.ndim==len(shape) and all(a is None or a==b for a,b in zip(shape,tensor.shape))
    if not ok and throw:raise ValueError('Tensor shape mismatch')
    return ok
