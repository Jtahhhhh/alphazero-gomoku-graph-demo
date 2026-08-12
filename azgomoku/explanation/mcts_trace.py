"""Root-only MCTS trace conversion; no leaf evidence or rendering."""

from .explanation_schema import cell


def extract_mcts_trace(root, selected_move, size, raw_policy_priors=None, top_k=5, playouts=None):
    if root is None: return {"available":False,"playouts":playouts,"selected":None,"top_candidates":[]}
    total=sum(child.n for child in root.children.values()); candidates=[]
    for action,child in root.children.items():
        item=cell(action,size); raw=None if raw_policy_priors is None else float(raw_policy_priors[action])
        item.update({"raw_policy_prior":raw,"search_prior":float(child.prior),"visits":int(child.n),"q":float(child.q),"pi":float(child.n/total) if total else 0.0,"selected":int(action)==int(selected_move)})
        candidates.append(item)
    candidates.sort(key=lambda x:(-x["visits"],-x["search_prior"],x["action"])); chosen=next((x for x in candidates if x["selected"]),None)
    return {"available":True,"playouts":playouts,"root_visits":int(total),"selected":chosen,"top_candidates":candidates[:top_k]}
