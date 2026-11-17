"""Dense-only TRELLIS utility compatibility (sparse CUDA imports removed)."""
from torch import nn
def zero_module(module):
    for p in module.parameters():p.detach().zero_()
    return module
def convert_module_to_f16(module):
    if isinstance(module,(nn.Conv3d,nn.Linear)):
        for p in module.parameters():p.data=p.data.half()
def convert_module_to_f32(module):
    if isinstance(module,(nn.Conv3d,nn.Linear)):
        for p in module.parameters():p.data=p.data.float()
