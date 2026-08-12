import torch
from models.cnn_baseline import CNNBaseline
from models.rgcn import RGCN
def test_shape():
    for model in (CNNBaseline(),RGCN()):
        p,v=model(torch.zeros(1,6,6,6)); assert p.shape==(1,36) and v.shape==(1,)
def test_fixed_dataset_overfit():
    m=CNNBaseline(hidden_dim=16); x=torch.zeros(4,6,6,6); target=torch.tensor([0,1,2,3]); opt=torch.optim.Adam(m.parameters(),lr=.02)
    first=None
    for i in range(20):
        p,v=m(x); loss=torch.nn.functional.cross_entropy(p,target)+(v**2).mean(); first=loss.item() if first is None else first; opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item()<first
