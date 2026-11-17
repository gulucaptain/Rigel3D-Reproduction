"""Data-only parsing of a narrowly whitelisted torch-save metadata format.

No external torch.load(weights_only=False), no arbitrary pickle global lookup.
Tensor reconstruction records metadata without loading tensor storage.
"""
from collections import OrderedDict
from dataclasses import dataclass
import io
import pickle
import numpy as np


@dataclass
class StorageRef:
    dtype: str
    key: str
    size: int


@dataclass
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple
    stride: tuple


def rebuild(storage,offset,shape,stride,requires_grad=False,backward_hooks=None,metadata=None):
    if not isinstance(storage,StorageRef):raise ValueError('Invalid storage')
    if not all(isinstance(i,int) and 0<=i<10**9 for i in (*shape,*stride,offset)):raise ValueError('Invalid tensor view')
    return TensorRef(storage,offset,tuple(shape),tuple(stride))


class MetadataReader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return OrderedDict
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
        if module=='torch' and name in ['FloatStorage','HalfStorage','CharStorage','IntStorage','LongStorage','ByteStorage','BoolStorage']:
            return dict(FloatStorage='<f4',HalfStorage='<f2',CharStorage='i1',IntStorage='<i4',LongStorage='<i8',ByteStorage='u1',BoolStorage='?')[name]
        raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')
    def persistent_load(self,pid):
        if len(pid)!=5 or pid[0]!='storage' or not isinstance(pid[1],str) or not str(pid[2]).isdigit():raise ValueError('Unsupported persistent ID')
        if pid[1] not in ['<f4','<f2','i1','<i4','<i8','u1','?'] or not isinstance(pid[4],int) or not 0<=pid[4]<=64*1024*1024:raise ValueError('Unsupported storage dtype/size')
        return StorageRef(pid[1],str(pid[2]),int(pid[4]))


def read_metadata(raw):return MetadataReader(io.BytesIO(raw)).load()


def materialize(ref,read_storage):
    if not isinstance(ref,TensorRef):return ref
    data=read_storage(ref.storage.key)
    dtype=np.dtype(ref.storage.dtype)
    if len(data)!=ref.storage.size*dtype.itemsize:raise ValueError('Storage length mismatch')
    if not ref.shape:return np.frombuffer(data,dtype=dtype)[ref.offset].copy()
    last=ref.offset+sum((n-1)*s for n,s in zip(ref.shape,ref.stride)) if all(ref.shape) else ref.offset
    if last>=ref.storage.size:raise ValueError('Tensor view exceeds storage')
    return np.ndarray(ref.shape,dtype=dtype,buffer=data,offset=ref.offset*dtype.itemsize,
        strides=tuple(s*dtype.itemsize for s in ref.stride)).copy()
