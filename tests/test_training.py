import io
import pickle
import numpy as np
import pytest
import torch
from rigel3d.decoders import SkeletonDecoder,skeleton_loss,RigDecoders
from rigel3d.torchzip import read_metadata,materialize
from rigel3d.metrics import skeleton_metrics,point_segment_distance,skinning_metrics


def test_data_only_torch_metadata_roundtrip():
    import zipfile
    tensor=torch.arange(24,dtype=torch.float32).reshape(4,6)[:,::2]
    f=io.BytesIO();torch.save([dict(name='safe',x=tensor)],f);f.seek(0)
    with zipfile.ZipFile(f) as z:
        name=next(n for n in z.namelist() if n.endswith('/data.pkl'));prefix=name[:-8]
        d=read_metadata(z.read(name))
        np.testing.assert_array_equal(materialize(d[0]['x'],lambda key:z.read(prefix+'data/'+key)),tensor.numpy())
    with pytest.raises(pickle.UnpicklingError):read_metadata(b'cos\nsystem\n.')


def test_causal_teacher_forcing_no_future_tuple_access():
    torch.manual_seed(2);m=SkeletonDecoder(width=16,heads=4,layers=1,max_joints=8).eval()
    s=torch.randn(1,4,16);r=torch.randn(1,3,16)
    t=torch.tensor([[[12,34,56,0],[14,36,58,0],[16,38,60,1],[18,40,62,2]]])
    a=m(t,s,r);u=t.clone();u[:,2:,:3]=(u[:,2:,:3]+13)%128;b=m(u,s,r)
    torch.testing.assert_close(a['coords'][:,:3],b['coords'][:,:3],atol=1e-6,rtol=1e-5)
    assert (a['coords'][:,3:]-b['coords'][:,3:]).abs().max()>1e-4


def test_valid_parent_mask_and_eos_training():
    m=SkeletonDecoder(width=16,heads=4,layers=1,max_joints=8)
    t=torch.tensor([[[12,34,56,0],[14,36,58,0],[16,38,60,1]]]);o=m(t,torch.randn(1,4,16),torch.randn(1,3,16))
    for k in range(4):assert (o['parent'][0,k,max(k,1):]==-1e4).all()
    loss,_=skeleton_loss(o,t,torch.tensor([3]));loss.backward();assert m.stop.weight.grad.abs().sum()>0


def test_masked_context_padding_invariance():
    torch.manual_seed(5);m=SkeletonDecoder(width=16,heads=4,layers=1,max_joints=8).eval()
    t=torch.tensor([[[12,34,56,0]]]);s=torch.randn(1,4,16);r=torch.randn(1,3,16);a=m(t,s,r)
    sp=torch.cat((s,torch.randn(1,2,16)*100),1);rp=torch.cat((r,torch.randn(1,2,16)*100),1)
    b=m(t,sp,rp,torch.tensor([[False]*4+[True]*2]),torch.tensor([[False]*3+[True]*2]))
    torch.testing.assert_close(a['coords'],b['coords'],atol=1e-6,rtol=1e-5)


def test_free_generation_ends_with_valid_tree_or_explicit_cap():
    m=SkeletonDecoder(width=16,heads=4,layers=1,max_joints=8)
    with torch.no_grad():m.stop.weight.zero_();m.stop.bias[:]=torch.tensor([0.,10.])
    t,d=m.generate(torch.randn(1,3,16),torch.randn(1,4,16))
    assert t.shape==(1,4) and t[0,3]==0 and d['eos']


def test_distance_and_weight_metrics_identity_and_degenerate_prediction():
    j=np.array([[0.,0.,0.],[1.,0.,0.]]);p=np.array([0,0])
    d=skeleton_metrics(j,p,j,p);assert max(d[k] for k in ['j2j','j2b','b2b'])<1e-12
    assert skeleton_metrics(j[:1],p[:1],j,p)['j2b']>0
    a=point_segment_distance(np.array([[.5,1,0],[2,0,0]]),j[:1],j[1:]);np.testing.assert_allclose(a[:,0],[1,1])
    w=np.array([[.1,.9],[.8,.2]]);s=skinning_metrics(w,w);assert s['l1_element']==0 and s['kl']==0 and s['precision']==1


def test_balanced_eos_loss_is_explicit_equal_class_mean():
    torch.manual_seed(10);m=SkeletonDecoder(width=16,heads=4,layers=1,max_joints=8)
    target=torch.tensor([[[1,2,3,0],[2,3,4,0],[3,4,5,1]]]);out=m(target,torch.randn(1,4,16),torch.randn(1,3,16))
    _,terms=skeleton_loss(out,target,torch.tensor([3]),balanced_stop=True)
    ce=torch.nn.functional.cross_entropy(out['stop'][0],torch.tensor([0,0,0,1]),reduction='none')
    torch.testing.assert_close(terms['stop'],.5*(ce[:3].mean()+ce[3]))


def test_sparse_skinning_correspondence_and_invalid_forest_preservation():
    from rigel3d.data import prepare_asset
    raw=dict(pc=np.array([[-.3,0,0,1,0,0],[.3,0,0,1,0,0]],dtype=np.float32),
        joints=np.array([[-.3,0,0],[.3,0,0]]),conns=np.array([0,1]),bones=np.array([[-.3,0,0,.3,0,0]]),
        skins_index=np.array([[0,-1],[-1,-1]]),skins_weight=np.array([[.7,0],[0,0]]))
    d,r=prepare_asset(raw,context_tokens=4)
    assert r['skeleton_status']=='excluded_from_skeleton' and d['weights'][0,0]==1 and not d['skin_valid'][1]
    assert d['tokens'].shape==(0,4)
