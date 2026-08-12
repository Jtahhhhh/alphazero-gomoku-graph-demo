import argparse
from pathlib import Path
from azgomoku.config import Config
from azgomoku.training import run_experiment

def main(name,model_cls):
    p=argparse.ArgumentParser(); p.add_argument("--profile",choices=("smoke","default","pilot"),default="default"); args=p.parse_args(); cfg=Config.profile(args.profile)
    model=model_cls(board_size=cfg.board_size,hidden_dim=cfg.hidden_dim,attention_heads=cfg.attention_heads)
    run_experiment(name,model,cfg,Path(__file__).parents[1]/"results"/name)
