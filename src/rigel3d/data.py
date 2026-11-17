"""Original Anymate adaptation. Targets and reconstruction context are explicit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .representation import bfs_tokens,rig_voxels


def geometric_features(pc):
    x=pc[:,:3];n=pc[:,3:]
    return np.concatenate((pc,x*x,n*n,(x*n).sum(-1,keepdims=True),np.linalg.norm(x,axis=-1,keepdims=True),np.ones((len(x),1)),np.zeros((len(x),1))),-1)


def fps_indices(points,n):
    n=min(len(points),n);chosen=[];distance=np.full(len(points),np.inf);idx=0
    for _ in range(n):
        chosen.append(idx);distance=np.minimum(distance,np.square(points-points[idx]).sum(-1));distance[chosen]=-1;idx=int(distance.argmax())
    return np.array(chosen)


def pool_cells(coords,features):
    unique,idx=np.unique(coords,axis=0,return_inverse=True);summed=np.zeros((len(unique),features.shape[1]));count=np.zeros(len(unique))
    np.add.at(summed,idx,features);np.add.at(count,idx,1)
    return unique,summed/count[:,None]


def prepare_asset(raw,context_tokens=48):
    pc=raw['pc'].astype(np.float64);joints=raw['joints'].astype(np.float64);bones=raw['bones'].astype(np.float64).reshape(-1,2,3)
    parents=raw['conns'].astype(int);nb=len(bones)
    if not all(np.isfinite(v).all() for v in [pc,joints,bones]):raise ValueError('Nonfinite raw geometry')
    # Unit box over all given geometry. This is a reconstruction convention,
    # explicitly differs from unknown author TRELLIS mesh normalization.
    union=np.concatenate((pc[:,:3],joints,bones.reshape(-1,3)))
    lo=union.min(0);hi=union.max(0);centre=(hi+lo)/2;scale=float((hi-lo).max())
    if scale<1e-6:raise ValueError('Degenerate asset')
    pc[:,:3]=(pc[:,:3]-centre)/scale;joints=(joints-centre)/scale;bones=(bones-centre)/scale
    indices=raw['skins_index'].astype(int);values=raw['skins_weight'].astype(np.float64)
    if len(indices)!=len(pc) or indices.shape!=values.shape:raise ValueError('Point/skinning correspondence mismatch')
    if (values<0).any() or (indices>=nb).any() or ((indices<0)&(values>0)).any():raise ValueError('Invalid sparse skinning')
    weights=np.zeros((len(pc),nb));valid=indices>=0
    np.add.at(weights,(np.broadcast_to(np.arange(len(pc))[:,None],indices.shape)[valid],indices[valid]),values[valid])
    mass=weights.sum(-1);good=mass>1e-8
    # Preserve indices: zero-mass points stay in source, excluded by explicit mask.
    weights[good]/=mass[good,None]
    feat=geometric_features(pc)
    cell=np.floor((pc[:,:3]+.5)*64).astype(int).clip(0,63)
    sc,sf=pool_cells(cell,feat);si=fps_indices((sc+.5)/64-.5,context_tokens)
    # This context uses GT skinning as Eq.2 prescribes: valid for reconstruction,
    # not image/mesh-only inference. A separate geometry-only control is saved.
    bone_feat=weights.T@feat/(weights.sum(0)[:,None]+1e-8)
    rc,rf,_=rig_voxels(bones[:,0],bones[:,1],bone_feat,64);ri=fps_indices((rc+.5)/64-.5,context_tokens)
    geometric_bone=geometric_features(np.concatenate((bones[:,0],bones[:,1]-bones[:,0]),-1))
    gc,gf,_=rig_voxels(bones[:,0],bones[:,1],geometric_bone,64)
    assert np.array_equal(gc,rc)
    status='eligible';reason=None;tokens=np.zeros((0,4),int);order=np.zeros(0,int)
    try:tokens,order=bfs_tokens(joints,parents)
    except ValueError as e:status='excluded_from_skeleton';reason=str(e)
    return (dict(points=pc[:,:3].astype('float32'),normals=pc[:,3:].astype('float32'),weights=weights.astype('float32'),skin_valid=good,
        joints=joints.astype('float32'),parents=parents,bones=bones.astype('float32'),tokens=tokens,order=order,
        surface_features=sf[si].astype('float32'),surface_coords=((sc[si]+.5)/64-.5).astype('float32'),
        rig_features=rf[ri].astype('float32'),geometry_rig_features=gf[ri].astype('float32'),rig_coords=((rc[ri]+.5)/64-.5).astype('float32')),
        dict(skeleton_status=status,skeleton_exclusion_reason=reason,joints=len(joints),bones=nb,skin_valid_points=int(good.sum()),
            skin_source_mass_min=float(mass.min()),skin_source_mass_max=float(mass.max()),source_mass_mean_abs_deviation=float(np.abs(mass-1).mean()),
            normalization=dict(centre=centre.tolist(),scale=scale,scope='Unit longest extent of union of original points, joints and bones; reconstruction inputs include GT rig'),
            surface_active_voxels=len(sc),rig_active_voxels=len(rc),surface_context_tokens=len(si),rig_context_tokens=len(ri)))


def load_assets(root):
    manifest=json.loads((root/'results/anymate_preparation.json').read_text());groups={s:[] for s in ['train','val','test']}
    for r in manifest['assets']:
        d=dict(np.load(root/'data/external/prepared'/r['filename']));d['record']=r;groups[r['split']].append(d)
    return groups
