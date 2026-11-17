import itertools
import numpy as np
import torch
from torch.nn import functional as F
from rigel3d.trellis_native import Sparse,SubMConv,window_sdpa,absolute_pe

def test_native_subm_conv_dense_reference_and_grad():
    torch.manual_seed(23);coords=np.array([[0,0,0],[1,0,0],[1,1,0],[3,2,1],[2,2,2]],np.int32);x=torch.randn(5,3,requires_grad=True);m=SubMConv(3,4)
    result=m(Sparse(x,coords)).feats;dense=torch.zeros(1,3,5,5,5);dense[0,:,coords[:,0],coords[:,1],coords[:,2]]=x.T
    expected=F.conv3d(dense,m.conv.weight.permute(0,4,1,2,3),m.conv.bias,padding=1)[0,:,coords[:,0],coords[:,1],coords[:,2]].T
    assert torch.allclose(result,expected,atol=2e-6)
    actual_grad=torch.autograd.grad(result.square().sum(),(x,m.conv.weight),retain_graph=True);ref_grad=torch.autograd.grad(expected.square().sum(),(x,m.conv.weight))
    for a,b in zip(actual_grad,ref_grad):assert torch.allclose(a,b,atol=3e-6)

def test_shifted_window_matches_direct_group_reference():
    torch.manual_seed(41);coords=np.array(list(itertools.product([0,3,7,8,12],repeat=3)));qkv=torch.randn(len(coords),3,2,8,requires_grad=True)
    for shift in [0,4]:
        result=window_sdpa(qkv,coords,shift);groups=(coords+shift)//8;oracle=torch.zeros_like(result)
        for key in np.unique(groups,axis=0):
            ids=np.flatnonzero((groups==key).all(-1));q,k,v=qkv[ids].permute(1,2,0,3).unbind(0);oracle[ids]=F.scaled_dot_product_attention(q,k,v).transpose(0,1).flatten(1)
        assert torch.allclose(result,oracle,atol=1e-6)
        result.square().sum().backward(retain_graph=True);assert torch.isfinite(qkv.grad).all()

def test_subdivision_order_and_ape():
    x=Sparse(torch.tensor([[2.,3.],[4.,5.]]),[[1,2,3],[3,2,1]]);y=x.subdivide();expected=np.array(list(itertools.product(range(2),repeat=3)))+np.array([2,4,6]);np.testing.assert_array_equal(y.coords[:8],expected);assert torch.equal(y.feats[:8],x.feats[0].expand(8,-1))
    pe=absolute_pe(np.array([[0,0,0],[1,2,3]]),768,'cpu');assert pe.shape==(2,768);assert torch.all(pe[0,:128]==0) and torch.all(pe[0,128:256]==1)

def test_occupancy_sampling_axis_order_and_gradient():
    from rigel3d.trellis_flow import sample_occ_features
    axes=torch.arange(16,dtype=torch.float32);x,y,z=torch.meshgrid(axes,axes,axes,indexing='ij');volume=(100*x+10*y+z)[None,None].requires_grad_();points=(torch.tensor([[3.,4.,5.],[11.,2.,9.]])+.5)/16-.5
    actual=sample_occ_features(volume,points);assert torch.allclose(actual[:,0],torch.tensor([345.,1129.]),atol=1e-3);actual.sum().backward();assert torch.allclose(volume.grad.sum(),torch.tensor(2.))

def test_coupled_flow_shapes_and_cross_branch_dependence():
    from rigel3d.trellis_flow import CoupledFlows
    torch.manual_seed(1);m=CoupledFlows();state=torch.randn(1,16,16,16,16,requires_grad=True);image=torch.randn(1,64,1024);time=torch.tensor([.4]);v=m.velocity(state,time,image);assert v.shape==state.shape
    v[:,:8].square().mean().backward();assert state.grad[:,8:].abs().sum()>0

def test_rig_adapter_preserves_pretrained_function_at_initialization():
    from rigel3d.joint_trellis import RigAdapters
    torch.manual_seed(2);m=RigAdapters();x=torch.randn(7,768);rig=(torch.randn(9,8),torch.randn(9,3));y=m(0,x,rig);assert torch.equal(x,y)
    y.square().mean().backward();assert m.output[0].weight.grad.abs().sum()>0

def test_image_runtime_budget_failure_never_silently_truncates():
    from rigel3d.trellis_runtime import generate_image_only
    class Flow:
        def sample_occupancy(self,image,steps,cfg):return torch.zeros(1,16,16,16,16)
    class Decoder:
        def __call__(self,z):return torch.ones(1,1,64,64,64)
    stats={m+'_'+k:np.full(8,0 if k.endswith('mean') else 1,np.float32) for m in ['surface','rig'] for k in ['occ_mean','occ_std']}
    result,status=generate_image_only(torch.zeros(1,64,1024),Flow(),None,Decoder(),stats)
    assert status['failure']=='surface_occupancy_budget_exceeded'
    assert result['surface_occupancy'].sum()==64**3 and status['surface_voxels']==64**3
    assert 'vertices' not in result
