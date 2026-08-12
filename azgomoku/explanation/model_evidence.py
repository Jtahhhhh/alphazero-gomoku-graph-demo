"""One explicit evidence-enabled root forward; never used by normal MCTS."""

import torch

from azgomoku.graph import RELATIONS, cell_graph

from .explanation_schema import cell


def model_type(model):
    return model.__class__.__name__.lower()


def collect_model_evidence(state, model, selected_move):
    device=next(model.parameters()).device
    x=torch.from_numpy(state.features()).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad(): logits,value,raw=model(x,return_evidence=True)
    masked=torch.full_like(logits,-torch.inf); legal=torch.as_tensor(state.legal_actions(),device=device); masked[0,legal]=logits[0,legal]
    priors=torch.softmax(masked,dim=-1)[0].cpu()
    name=model_type(model); edges=[]; semantic={}
    if name=="rgcn":
        for relation_index,(relation,group) in enumerate(zip(RELATIONS,cell_graph(state.size))):
            for source,target in group.t().tolist():
                edges.append({"relation":relation,"source":cell(source,state.size),"target":cell(target,state.size),"attention":None,"head_attention":None})
    elif name=="rgat":
        for relation_index,relation in enumerate(RELATIONS):
            group=getattr(model,f"edge_{relation_index}").t().cpu().tolist(); alpha=raw["relation_attention"][relation][0].cpu()
            for index,(source,target) in enumerate(group):
                heads=[float(v) for v in alpha[index].tolist()]
                edges.append({"relation":relation,"source":cell(source,state.size),"target":cell(target,state.size),"head_attention":heads,"attention":sum(heads)/len(heads),"attention_aggregation":"mean across attention heads","layer":"final"})
    elif name=="han":
        for relation_index,relation in enumerate(RELATIONS):
            group=getattr(model,f"meta_{relation_index}").t().cpu().tolist(); alpha=raw["node_attention"][relation][0].cpu()
            for index,(source,target) in enumerate(group):
                edges.append({"relation":relation,"source":cell(source,state.size),"target":cell(target,state.size),"attention":float(alpha[index]),"head_attention":None,"meta_path":relation})
        semantic={relation:float(weight) for relation,weight in zip(RELATIONS,raw["semantic_attention"].cpu().tolist())}
    graph={"attention_available":bool(raw["attention_available"]),"evidence_kind":"structural_relations" if name=="rgcn" else "learned_attention","edges":edges}
    limitations=[]
    if name=="rgcn": limitations.append("R-GCN exposes structural relations but no learned attention coefficients.")
    return {
        "model_type":name,
        "network":{"value":float(value.item()),"raw_policy_prior":float(priors[int(selected_move)]),"raw_policy_priors":[float(v) for v in priors.tolist()]},
        "graph_evidence":graph,"semantic_attention":semantic,"limitations":limitations,
    }
