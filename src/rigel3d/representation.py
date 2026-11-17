"""Paper Eq.2, bone occupancy, BFS joint tokens. Explicit boundary conventions."""
import numpy as np
import torch


def pool_bone_features(vertex_features,weights,epsilon=1e-8):
    if (weights<0).any():raise ValueError('Negative skinning weight')
    return weights.T@vertex_features/(weights.sum(0)[:,None]+epsilon)


def segment_voxels(head,tail,resolution=64):
    """All CLOSED cells touched by a segment in [-.5,.5]^3.

    Authors do not disclose the boundary convention. Touches at faces/edges
    count; no clipping of out-of-domain geometry is silently performed.
    """
    a=np.asarray(head,dtype=float);b=np.asarray(tail,dtype=float)
    if min(a.min(),b.min())<-.5-1e-10 or max(a.max(),b.max())>.5+1e-10:raise ValueError('Normalize geometry and skeleton together')
    if resolution<2:raise ValueError('Invalid resolution')
    a=(a+.5)*resolution;b=(b+.5)*resolution
    lower=np.maximum(np.ceil(np.minimum(a,b)).astype(int)-1,0)
    upper=np.minimum(np.floor(np.maximum(a,b)).astype(int),resolution-1)
    cells=np.stack(np.meshgrid(*[np.arange(lo,hi+1) for lo,hi in zip(lower,upper)],indexing='ij'),-1).reshape(-1,3)
    lo=np.zeros(len(cells));hi=np.ones(len(cells));valid=np.ones(len(cells),bool);delta=b-a
    for j in range(3):
        if abs(delta[j])<1e-12:valid&=(a[j]>=cells[:,j]-1e-10)&(a[j]<=cells[:,j]+1+1e-10)
        else:
            t0=(cells[:,j]-a[j])/delta[j];t1=(cells[:,j]+1-a[j])/delta[j]
            lo=np.maximum(lo,np.minimum(t0,t1));hi=np.minimum(hi,np.maximum(t0,t1))
    return cells[valid&(lo<=hi+1e-10)]


def rig_voxels(heads,tails,bone_features,resolution=64):
    """Average distinct bone features where their segment occupancies overlap."""
    coords=[];features=[]
    for a,b,f in zip(heads,tails,bone_features):
        c=segment_voxels(a,b,resolution);coords.append(c);features.append(np.repeat(np.asarray(f)[None],len(c),axis=0))
    if not coords:raise ValueError('Empty rig')
    unique,idx=np.unique(np.concatenate(coords),axis=0,return_inverse=True)
    feat=np.concatenate(features);summed=np.zeros((len(unique),feat.shape[1]));count=np.zeros(len(unique))
    np.add.at(summed,idx,feat);np.add.at(count,idx,1)
    return unique,summed/count[:,None],count.astype(int)


def bfs_tokens(joints,parents,resolution=128):
    """Sort each BFS DEPTH globally by z,y,x; root parent points to itself.

    Quantization uses nearest grid endpoints: round((x+.5)*(R-1)).
    This rounding and exact-coordinate tie rule are explicit assumptions.
    """
    joints=np.asarray(joints,dtype=float);parents=np.asarray(parents,dtype=int);n=len(joints)
    if parents.shape!=(n,) or n==0 or (parents<0).any() or (parents>=n).any():raise ValueError('Invalid parent indices')
    roots=np.flatnonzero(parents==np.arange(n))
    if len(roots)!=1:raise ValueError('Expected one root')
    order=[];level=roots.tolist();seen=set()
    while level:
        level=sorted(level,key=lambda i:(joints[i,2],joints[i,1],joints[i,0],i))
        if any(i in seen for i in level):raise ValueError('Cycle')
        seen.update(level);order.extend(level)
        level=[int(i) for i in range(n) if parents[i] in level and parents[i]!=i]
    if len(order)!=n:raise ValueError('Disconnected cycle')
    if (joints<-.5).any() or (joints>.5).any():raise ValueError('Joint outside normalized domain')
    order=np.array(order);inverse=np.empty(n,int);inverse[order]=np.arange(n)
    xyz=np.rint((joints[order]+.5)*(resolution-1)).astype(int)
    return np.column_stack((xyz,inverse[parents[order]])),order


def decode_tokens(tokens,resolution=128):
    t=np.asarray(tokens)
    if t.ndim!=2 or t.shape[1]!=4 or len(t)==0:raise ValueError('Invalid shape')
    if (t[:,:3]<0).any() or (t[:,:3]>=resolution).any():raise ValueError('Invalid coordinate')
    if t[0,3]!=0 or (t[1:,3]<0).any() or (t[1:,3]>=np.arange(1,len(t))).any():raise ValueError('Invalid parent topology')
    return t[:,:3]/(resolution-1)-.5,t[:,3].astype(int)
