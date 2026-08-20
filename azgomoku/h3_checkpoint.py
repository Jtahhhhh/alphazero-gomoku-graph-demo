"""Immutable, resumable checkpoint bundles for controlled H3 pilot runs."""

import json
from pathlib import Path
import torch

from .reproducibility import restore_rng_state, rng_state


FORMAT_VERSION=1


def make_bundle(model_type,model,optimizer,replay,config,training_state):
    return {
        "format_version":FORMAT_VERSION,"model_type":model_type,
        "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
        "training_state":dict(training_state),"config":dict(config),"seed":int(config["seed"]),
        "rng_state":rng_state(),"replay_snapshot":{"capacity":replay.data.maxlen,"samples":list(replay.data)},
    }


def save_bundle(run_dir,bundle):
    iteration=int(bundle["training_state"]["iteration"]); checkpoint_dir=Path(run_dir)/"checkpoints"; checkpoint_dir.mkdir(parents=True,exist_ok=True)
    path=checkpoint_dir/f"iter_{iteration:03d}.pt"
    if path.exists(): raise FileExistsError(f"checkpoint is immutable: {path}")
    torch.save(bundle,path)
    manifest_path=checkpoint_dir/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"format_version":FORMAT_VERSION,"model_type":bundle["model_type"],"seed":bundle["seed"],"checkpoints":[]}
    manifest["checkpoints"].append({"iteration":iteration,"path":path.name,"training_state":bundle["training_state"],"bytes":path.stat().st_size})
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return path


def load_bundle(path,model,optimizer,replay,expected_model_type=None,device=None):
    """Restore a bundle without changing its on-disk schema.

    ``device`` is a runtime concern: old CPU checkpoints can be resumed on a
    GPU and GPU checkpoints can be loaded on a CPU.
    """
    # Always deserialize checkpoint tensors onto CPU first so the saved CPU RNG
    # state stays a CPU ByteTensor for restore_rng_state(). Model/optimizer
    # tensors are moved to ``device`` after loading when requested.
    bundle=torch.load(path,map_location="cpu",weights_only=False)
    if bundle.get("format_version")!=FORMAT_VERSION: raise ValueError("unsupported checkpoint format")
    if expected_model_type and bundle["model_type"]!=expected_model_type: raise ValueError("checkpoint model type mismatch")
    model.load_state_dict(bundle["model_state"]); optimizer.load_state_dict(bundle["optimizer_state"])
    if device is not None:
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor): state[key]=value.to(device)
    snapshot=bundle["replay_snapshot"]; replay.data.clear(); replay.data.extend(snapshot["samples"])
    if replay.data.maxlen!=snapshot["capacity"]: raise ValueError("replay capacity mismatch")
    restore_rng_state(bundle["rng_state"])
    return bundle


def model_from_bundle(path,model_classes):
    bundle=torch.load(path,map_location="cpu",weights_only=False); config=bundle["config"]; model_type=bundle["model_type"]
    model=model_classes[model_type](board_size=config["board_size"],hidden_dim=config["hidden_dim"],attention_heads=config["attention_heads"])
    model.load_state_dict(bundle["model_state"]); model.eval(); return model,bundle
