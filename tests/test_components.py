import numpy as np
import pytest
import torch
from rigel3d.representation import pool_bone_features,segment_voxels,rig_voxels,bfs_tokens,decode_tokens
from rigel3d.skinning import SkinningDecoder,soft_cross_entropy


def test_weighted_bone_pooling_and_empty_bone_gradient():
    f=torch.tensor([[1.,2.],[4.,8.]],dtype=torch.float64,requires_grad=True)
    w=torch.tensor([[.75,.25,0.],[.25,.75,0.]],dtype=torch.float64)
    y=pool_bone_features(f,w)
    torch.testing.assert_close(y,torch.tensor([[1.75,3.5],[3.25,6.5],[0.,0.]],dtype=torch.float64))
    y.sum().backward();assert torch.isfinite(f.grad).all()


def test_segment_boundary_cells_and_reversal():
    # A horizontal line through cell interiors hits four cells; endpoints touch
    # the first/last outer domain boundary without generating invalid cells.
    expected=np.array([[i,0,0] for i in range(4)])
    actual=segment_voxels([-.5,-.4,-.4],[.5,-.4,-.4],4)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual,segment_voxels([.5,-.4,-.4],[-.5,-.4,-.4],4))
    assert len(segment_voxels([0,0,0],[0,0,0],4))==8
    with pytest.raises(ValueError):segment_voxels([-.6,0,0],[0,0,0])


def test_overlapping_bones_average_distinct_features():
    c,f,count=rig_voxels(np.array([[-.4]*3,[-.4]*3]),np.array([[-.3]*3,[-.3]*3]),np.array([[1.,0.],[0.,3.]]),4)
    np.testing.assert_allclose(f,np.tile([.5,1.5],(len(c),1)))
    np.testing.assert_array_equal(count,np.full(len(c),2))


def test_bfs_global_depth_order_and_quantization_bound():
    j=np.array([[0,0,0],[.2,.2,.2],[-.2,-.2,-.2],[.4,.3,-.3],[-.4,.3,.3]])
    p=np.array([0,0,0,1,2]);t,order=bfs_tokens(j,p)
    assert order.tolist()==[0,2,1,3,4]
    recovered,parent=decode_tokens(t)
    assert np.max(np.linalg.norm(recovered-j[order],axis=-1))<=np.sqrt(3)/(2*127)+1e-12
    assert (parent[1:]<np.arange(1,len(j))).all()


def test_bfs_permutation_invariant_for_unique_coordinates():
    rng=np.random.default_rng(2026)
    for _ in range(100):
        n=int(rng.integers(2,50));j=rng.uniform(-.5,.5,(n,3));p=np.array([0]+[rng.integers(i) for i in range(1,n)])
        t,_=bfs_tokens(j,p);perm=rng.permutation(n);inv=np.argsort(perm)
        u,_=bfs_tokens(j[perm],inv[p[perm]])
        np.testing.assert_array_equal(t,u)


def test_invalid_tree_and_forward_parent_rejected():
    with pytest.raises(ValueError):bfs_tokens(np.zeros((3,3)),[0,2,1])
    with pytest.raises(ValueError):decode_tokens(np.array([[0,0,0,0],[1,1,1,1]]))


def inputs():
    torch.manual_seed(7)
    return [torch.randn(1,n,d) for n,d in [(12,3),(12,3),(5,3),(5,3),(8,16),(6,16)]]


def test_skinning_probability_and_bone_permutation():
    model=SkinningDecoder().eval();x=inputs();w=model(*x)
    torch.testing.assert_close(w.sum(-1),torch.ones(1,12))
    perm=torch.tensor([2,0,4,1,3]);y=list(x);y[2]=x[2][:,perm];y[3]=x[3][:,perm]
    torch.testing.assert_close(model(*y),w[:,:,perm],atol=1e-6,rtol=1e-5)


def test_skinning_gradients_reach_both_latent_sets():
    model=SkinningDecoder();x=inputs();x[-1].requires_grad_();x[-2].requires_grad_()
    w=model(*x);target=torch.rand_like(w).softmax(-1);soft_cross_entropy(w,target).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert x[-1].grad.abs().sum()>0 and x[-2].grad.abs().sum()>0
