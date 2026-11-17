"""Explicit local metrics; no same-index comparison of generated bones."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def point_segment_distance(points,heads,tails):
    delta=tails-heads;offset=points[:,None]-heads[None]
    t=np.clip((offset*delta[None]).sum(-1)/np.maximum((delta*delta).sum(-1),1e-15),0,1)
    return np.linalg.norm(offset-t[...,None]*delta[None],axis=-1)


def skeleton_metrics(pred_joints,pred_parents,joints,parents,bone_samples=32):
    pred_joints=np.asarray(pred_joints);joints=np.asarray(joints)
    p=np.arange(len(pred_joints))!=pred_parents;g=np.arange(len(joints))!=parents
    pb=np.stack((pred_joints[pred_parents[p]],pred_joints[p]),1)
    gb=np.stack((joints[parents[g]],joints[g]),1)
    # A root-only failed prediction has no bones. Retain its joint as a
    # degenerate segment instead of discarding the failed case.
    if not len(pb):pb=np.repeat(pred_joints[:1,None],2,axis=1)
    if not len(gb):gb=np.repeat(joints[:1,None],2,axis=1)
    dist=cdist(pred_joints,joints)
    j2j=.5*(dist.min(0).mean()+dist.min(1).mean())
    j2b=.5*(point_segment_distance(pred_joints,gb[:,0],gb[:,1]).min(-1).mean()+point_segment_distance(joints,pb[:,0],pb[:,1]).min(-1).mean())
    t=np.linspace(0,1,bone_samples)
    ps=(pb[:,0,None]*(1-t[None,:,None])+pb[:,1,None]*t[None,:,None]).reshape(-1,3)
    gs=(gb[:,0,None]*(1-t[None,:,None])+gb[:,1,None]*t[None,:,None]).reshape(-1,3)
    # Quadrature of continuous point-to-segment distances, not endpoint-only CD.
    b2b=.5*(point_segment_distance(ps,gb[:,0],gb[:,1]).min(-1).mean()+point_segment_distance(gs,pb[:,0],pb[:,1]).min(-1).mean())
    return dict(j2j=float(j2j),j2b=float(j2b),b2b=float(b2b),joint_count=len(pred_joints),joint_count_abs_error=abs(len(pred_joints)-len(joints)))


def skinning_metrics(pred,target,threshold=.05):
    pred=np.asarray(pred,dtype=float);target=np.asarray(target,dtype=float)
    a=pred>=threshold;b=target>=threshold;tp=(a&b).sum(-1)
    # Official Anymate macro per-point precision/recall: zero if either set empty.
    valid=(a.sum(-1)>0)&(b.sum(-1)>0)
    precision=np.where(valid,tp/np.maximum(a.sum(-1),1),0).mean()
    recall=np.where(valid,tp/np.maximum(b.sum(-1),1),0).mean()
    return dict(l1_element=float(np.abs(pred-target).mean()),l1_vector=float(np.abs(pred-target).sum(-1).mean()),
        l2_element=float(np.square(pred-target).mean()),l2_vector=float(np.linalg.norm(pred-target,axis=-1).mean()),kl=float((target*np.log(np.maximum(target,1e-12)/np.maximum(pred,1e-12))).sum(-1).mean()),
        precision=float(precision),recall=float(recall))


def rignet_metrics(pred_joints,pred_parents,joints,parents,step=.005):
    """Independent implementation of pinned RigNet sampling/Chamfer convention.

    Length-weighted samples at round(bone_length/.005) intervals; distances
    unsquared, symmetric mean divided by two. Root-only failure retained.
    """
    def sample(j,p):
        result=[]
        for i in range(len(j)):
            if p[i]==i:continue
            a=j[p[i]];b=j[i];n=int(np.round(np.linalg.norm(b-a)/step))
            result.append(a[None]+(b-a)[None]*np.arange(n+1)[:,None]/(n+1e-30))
        return np.concatenate(result) if result else j[:1]
    ps=sample(pred_joints,pred_parents);gs=sample(joints,parents)
    def chamfer(a,b):return .5*(cKDTree(b).query(a)[0].mean()+cKDTree(a).query(b)[0].mean())
    return dict(j2j=float(chamfer(pred_joints,joints)),j2b=float(.5*(cKDTree(gs).query(pred_joints)[0].mean()+cKDTree(ps).query(joints)[0].mean())),
        b2b=float(chamfer(ps,gs)),step=step)
