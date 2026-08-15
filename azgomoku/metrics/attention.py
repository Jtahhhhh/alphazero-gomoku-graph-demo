"""Attention-collapse diagnostics shared by evaluation pipelines."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .semantic_alignment import entropy


def collapse_metrics(edges: list[dict]) -> dict:
    if not edges or edges[0].get("head_attention") is None:
        return {}
    grouped = defaultdict(list)
    for edge in edges:
        grouped[(edge["relation"], int(edge["target"]["action"]))].append(edge)
    entropies = []
    deviations = []
    for group in grouped.values():
        degree = len(group)
        heads = len(group[0]["head_attention"])
        if degree > 1:
            for head in range(heads):
                values = np.asarray([edge["head_attention"][head] for edge in group], dtype=float)
                values = values / (values.sum() or 1.0)
                entropies.append(entropy(values) / math.log(degree))
        deviations.extend(abs(float(edge["attention"]) - 1.0 / degree) for edge in group)
    attention = np.asarray([float(edge["attention"]) for edge in edges])
    structural = np.asarray([1.0 / len(grouped[(edge["relation"], int(edge["target"]["action"]))]) for edge in edges])
    correlation = 0.0
    if np.std(attention) > 0 and np.std(structural) > 0:
        correlation = float(np.corrcoef(attention, structural)[0, 1])
    heads = len(edges[0]["head_attention"])
    vectors = [np.asarray([edge["head_attention"][head] for edge in edges]) for head in range(heads)]
    distances = [float(np.mean(np.abs(vectors[i] - vectors[j]))) for i in range(heads) for j in range(i + 1, heads)]
    normalized_entropy = float(np.mean(entropies)) if entropies else 1.0
    structural_mae = float(np.mean(deviations)) if deviations else 0.0
    head_diversity = float(np.mean(distances)) if distances else 0.0
    collapse = normalized_entropy >= 0.98 and structural_mae <= 0.02 and head_diversity <= 0.02
    return {
        "attention_normalized_entropy": normalized_entropy,
        "attention_structural_mae": structural_mae,
        "attention_head_diversity": head_diversity,
        "attention_topology_correlation": correlation,
        "attention_collapse_flag": int(collapse),
    }
