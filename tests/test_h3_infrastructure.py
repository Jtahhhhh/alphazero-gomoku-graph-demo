import random
import numpy as np
import torch

from azgomoku.h3_checkpoint import load_bundle,make_bundle,model_from_bundle,save_bundle
from azgomoku.replay import ReplayBuffer
from azgomoku.reproducibility import rng_state,seed_everything
from models.rgcn import RGCN
from models.rgat import RGAT


def flattened(model): return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def build(seed,model_class=RGCN):
    seed_everything(seed); return model_class(board_size=6,hidden_dim=8,attention_heads=2)


def test_seed_before_initialization_is_reproducible_and_distinct():
    assert torch.equal(flattened(build(7)),flattened(build(7)))
    assert not torch.equal(flattened(build(7)),flattened(build(17)))


def test_checkpoint_restores_model_optimizer_counters_replay_and_rng(tmp_path):
    model=build(7,RGAT); optimizer=torch.optim.Adam(model.parameters(),lr=.01); replay=ReplayBuffer(8)
    x=torch.zeros(1,6,6,6); logits,value=model(x); loss=logits.mean()+value.mean(); loss.backward(); optimizer.step()
    replay.extend([("sample",1)]); config={"seed":7,"board_size":6,"hidden_dim":8,"attention_heads":2}; training={"iteration":5,"selfplay_games_seen":100,"optimizer_updates":40,"replay_size":1}
    bundle=make_bundle("rgat",model,optimizer,replay,config,training); expected_python=random.random(); expected_numpy=float(np.random.random()); expected_torch=float(torch.rand(()))
    path=save_bundle(tmp_path,bundle)
    restored=build(999,RGAT); restored_optimizer=torch.optim.Adam(restored.parameters(),lr=.5); restored_replay=ReplayBuffer(8)
    loaded=load_bundle(path,restored,restored_optimizer,restored_replay,"rgat")
    assert torch.equal(flattened(model),flattened(restored)); assert loaded["training_state"]==training
    assert restored_optimizer.state_dict()["state"] and list(restored_replay.data)==[("sample",1)]
    assert random.random()==expected_python and float(np.random.random())==expected_numpy and float(torch.rand(()))==expected_torch
    reconstructed,reloaded=model_from_bundle(path,{"rgat":RGAT}); assert torch.equal(flattened(model),flattened(reconstructed)) and reloaded["model_type"]=="rgat"


def test_checkpoint_names_are_immutable(tmp_path):
    model=build(7); optimizer=torch.optim.Adam(model.parameters()); replay=ReplayBuffer(2); config={"seed":7,"board_size":6,"hidden_dim":8,"attention_heads":2}; state={"iteration":0,"selfplay_games_seen":0,"optimizer_updates":0,"replay_size":0}
    bundle=make_bundle("rgcn",model,optimizer,replay,config,state); save_bundle(tmp_path,bundle)
    try: save_bundle(tmp_path,bundle)
    except FileExistsError: pass
    else: raise AssertionError("immutable checkpoint was overwritten")
