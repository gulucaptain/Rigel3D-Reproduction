import numpy as np
import torch
from rigel3d.sparse_vae import SparseConv,SparseVAE,neighbor_indices,OFFSETS
from rigel3d.flexicubes import FlexiCubes,regular_grid

def test_sparse_convolution_dense_oracle_and_gradient():
    coords=np.array([[0,0,0],[1,0,0],[1,1,0],[3,0,0]])
    n=torch.tensor(neighbor_indices(coords));conv=SparseConv(3);x=torch.randn(4,3,requires_grad=True);out=conv(x,n)
    oracle=[]
    for c in coords:
        v=conv.bias.clone()
        for k,o in enumerate(OFFSETS):
            found=np.flatnonzero((coords==c+o).all(-1))
            if len(found):v=v+x[int(found[0])]@conv.weight[k]
        oracle.append(v)
    assert torch.allclose(out,torch.stack(oracle),atol=1e-6)
    out.square().sum().backward();assert torch.isfinite(x.grad).all()

def test_sparse_vae_kl_and_stochastic_train():
    torch.manual_seed(4);v=SparseVAE(channels=5,width=8,latent=2);x=torch.randn(3,5);n=torch.tensor(neighbor_indices(np.array([[0,0,0],[1,0,0],[2,0,0]])))
    a,kl,mu=v(x,n);b,_,_=v(x,n);assert not torch.equal(a,b);assert kl>=0
    c,_,_=v(x,n,False);d,_,_=v(x,n,False);assert torch.equal(c,d)
    (a.square().mean()+kl).backward();assert v.stats.weight.grad.abs().sum()>0

def test_flexicubes_sphere_and_backward():
    verts,cubes=regular_grid(8);sdf=(verts.norm(dim=-1)-.3).requires_grad_();v,f,reg,_=FlexiCubes('cpu')(verts,sdf,cubes,8,training=True)
    assert len(v)>0 and len(f)>0 and int(f.max())<len(v)
    assert abs(float(v.norm(dim=-1).mean().detach())-.3)<.04
    (v.square().mean()+reg.mean()).backward();assert torch.isfinite(sdf.grad).all() and sdf.grad.abs().sum()>0

def test_image_generator_api_has_no_target_and_positions_follow_noise():
    import inspect
    from rigel3d.generation import FourFlows
    from rigel3d.inference import generate_asset
    assert list(inspect.signature(generate_asset).parameters)==['image','flows','field','decoders','latent_stats','steps','cfg']
    torch.manual_seed(5);m=FourFlows();image=torch.zeros(1,384);torch.manual_seed(10);a,_=m.sample(image,steps=2,max_cells=5)
    torch.manual_seed(10);b,_=m.sample(image,steps=2,max_cells=5)
    for k in ['surface','rig']:
        assert torch.equal(a[k]['coords'],b[k]['coords'])
        assert torch.equal(a[k]['features'],b[k]['features'])
        assert (a[k]['coords'].abs()<.5).all()
    torch.manual_seed(11);c,_=m.sample(image,steps=2,max_cells=5)
    assert not torch.equal(a['surface']['features'],c['surface']['features'])

def test_field_decoder_query_partition_invariant():
    from rigel3d.generation import FieldDecoder
    torch.manual_seed(3);m=FieldDecoder().eval();q=torch.randn(1,17,3);f=torch.randn(1,6,8);c=torch.randn(1,6,3)
    with torch.no_grad():
        whole=m(q,f,c,f,c);parts=torch.cat([m(x,f,c,f,c) for x in q.split(5,dim=1)],dim=1)
    assert torch.allclose(whole,parts,atol=1e-6)
