from torch import nn
class CNNBaseline(nn.Module):
    def __init__(self,board_size=6,hidden_dim=64,**_):
        super().__init__(); self.board_size=board_size
        self.body=nn.Sequential(nn.Conv2d(6,hidden_dim,3,padding=1),nn.ReLU(),nn.Conv2d(hidden_dim,hidden_dim,3,padding=1),nn.ReLU())
        self.policy=nn.Conv2d(hidden_dim,1,1); self.value=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1),nn.Tanh())
    def forward(self,x,return_evidence=False):
        h=self.body(x); outputs=(self.policy(h).flatten(1),self.value(h).squeeze(-1))
        return (*outputs,{"attention_available":False}) if return_evidence else outputs
