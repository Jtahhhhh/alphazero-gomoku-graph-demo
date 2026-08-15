"""Deterministic proof-to-graph semantic alignment metrics."""

from __future__ import annotations

import hashlib

import numpy as np


def entropy(probabilities):
    p=np.asarray(probabilities,dtype=float); p=p[p>0]
    return float(-(p*np.log(p)).sum())


def average_precision(labels,scores):
    positives=sum(labels)
    if not positives: return 0.0
    order=sorted(range(len(scores)),key=lambda i:(-scores[i],i)); hits=0; total=0.0
    for rank,index in enumerate(order,1):
        if labels[index]: hits+=1; total+=hits/rank
    return total/positives


def critical_ids(edges,proof):
    windows=[set(window) for window in proof["windows"]]; relations=set(proof["critical_relations"])
    return {edge["edge_id"] for edge in edges if edge["relation"] in relations and any(edge["source"]["action"] in window and edge["target"]["action"] in window for window in windows)}


def score_alignment(edges,scores,proof):
    critical=critical_ids(edges,proof); labels=[edge["edge_id"] in critical for edge in edges]
    total=sum(max(0,float(score)) for score in scores); mass=sum(max(0,float(score)) for score,label in zip(scores,labels) if label)/(total or 1)
    k=max(1,sum(labels)); order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),edges[i]["edge_id"])); top=order[:k]; hits=sum(labels[i] for i in top)
    return {"mass":mass,"precision_at_k":hits/k,"recall_at_k":hits/(sum(labels) or 1),"auprc":average_precision(labels,list(map(float,scores))),"critical_edges":len(critical)}


def aggregate_proofs(edges,scores,proofs):
    values=[score_alignment(edges,scores,proof) for proof in proofs]
    if not values: return {"best":{},"mean":{}}
    return {"best":max(values,key=lambda x:x["mass"]),"mean":{key:float(np.mean([v[key] for v in values])) for key in ("mass","precision_at_k","recall_at_k","auprc","critical_edges")}}


def baselines(edges,proofs,state_id):
    indegree={}
    for edge in edges: indegree[(edge["relation"],edge["target"]["action"])]=indegree.get((edge["relation"],edge["target"]["action"]),0)+1
    structural=[1/indegree[(edge["relation"],edge["target"]["action"])] for edge in edges]
    structural_result=aggregate_proofs(edges,structural,proofs)["mean"]
    seed=int(hashlib.sha256(state_id.encode()).hexdigest()[:8],16); rng=np.random.default_rng(seed); random_results=[]
    for _ in range(32): random_results.append(aggregate_proofs(edges,rng.random(len(edges)),proofs)["mean"])
    random_mean={key:float(np.mean([result[key] for result in random_results])) for key in ("mass","precision_at_k","recall_at_k","auprc")}
    return random_mean,structural_result
