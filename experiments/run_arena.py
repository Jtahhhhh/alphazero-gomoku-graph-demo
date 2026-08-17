"""Run balanced CNN/R-GCN/R-GAT Arena evaluation against heuristic agents."""

import argparse
from pathlib import Path

from investigation.arena import load_models, run_arena


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--rgcn-checkpoint", type=Path, required=True)
    parser.add_argument("--rgat-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument("--mcts-playouts", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-iteration", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
    models = load_models({"cnn_baseline": args.cnn_checkpoint, "rgcn": args.rgcn_checkpoint, "rgat": args.rgat_checkpoint}, args.device)
    run_arena(models, args.output, games=args.games, board_size=args.board_size, win_length=args.win_length, mcts_playouts=args.mcts_playouts, epsilon=args.epsilon, depth=args.depth, seed=args.seed, checkpoint_iteration=args.checkpoint_iteration)


if __name__ == "__main__":
    main()
