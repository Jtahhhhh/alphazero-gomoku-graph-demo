import math
import numpy as np
import torch

class Node:
    def __init__(self,prior=1.0): self.prior=prior; self.n=0; self.w=0.; self.children={}
    @property
    def q(self): return self.w/self.n if self.n else 0.

def predict(model,state):
    device=next(model.parameters()).device
    x=torch.from_numpy(state.features()).unsqueeze(0).to(device)
    with torch.no_grad(): logits,value=model(x,return_evidence=False)
    legal=state.legal_actions(); masked=torch.full_like(logits,-torch.inf); masked[0,legal]=logits[0,legal]
    probs=torch.softmax(masked,dim=-1)[0].cpu().numpy()
    return probs,float(value.item())

def search(model,state,playouts=50,c_puct=1.5,temperature=1.0,return_root=False):
    root=Node(); priors,_=predict(model,state)
    root.children={int(a):Node(float(priors[a])) for a in state.legal_actions()}
    for _ in range(playouts):
        node=root; s=state; path=[]
        while node.children:
            action,child=max(node.children.items(),key=lambda kv: kv[1].q+c_puct*kv[1].prior*math.sqrt(node.n+1)/(1+kv[1].n))
            path.append((node,child)); s=s.play(action); node=child
            if s.terminal(): break
        if s.terminal(): value=s.outcome_for(s.to_play)
        else:
            priors,value=predict(model,s); node.children={int(a):Node(float(priors[a])) for a in s.legal_actions()}
        for _,child in reversed(path): child.n+=1; child.w+=value; value=-value
        root.n+=1
    visits=np.zeros(state.size**2,np.float32)
    for a,c in root.children.items(): visits[a]=c.n
    if visits.sum()==0: visits[state.legal_actions()]=1
    if temperature<=1e-6:
        pi=np.zeros_like(visits); pi[int(visits.argmax())]=1
    else:
        visits=visits**(1/temperature); pi=visits/visits.sum()
    return (pi,root) if return_root else pi
