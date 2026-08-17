#!/usr/bin/env python3
"""Quick import test to check for syntax and import errors."""

import sys

print("Testing imports...")

try:
    from azgomoku.game import GomokuState
    print("✓ azgomoku.game.GomokuState")
except Exception as e:
    print(f"✗ azgomoku.game.GomokuState: {e}")
    sys.exit(1)

try:
    from azgomoku.agents import evaluate_position, AlphaBetaAgent, EGreedyAgent
    print("✓ azgomoku.agents (evaluate_position, AlphaBetaAgent, EGreedyAgent)")
except Exception as e:
    print(f"✗ azgomoku.agents: {e}")
    sys.exit(1)

try:
    from azgomoku.tracking.match_tracker import check_milestone, GameRecord, JsonGameLogger, ExcelGameLogger
    print("✓ azgomoku.tracking.match_tracker")
except Exception as e:
    print(f"✗ azgomoku.tracking.match_tracker: {e}")
    sys.exit(1)

try:
    from investigation.eval_harness import run_evaluation, EvalResult
    print("✓ investigation.eval_harness")
except Exception as e:
    print(f"✗ investigation.eval_harness: {e}")
    sys.exit(1)

try:
    from investigation.plot_dashboard import load_training_logs, plot_training_loss
    print("✓ investigation.plot_dashboard")
except Exception as e:
    print(f"✗ investigation.plot_dashboard: {e}")
    sys.exit(1)

try:
    from models.cnn_baseline import CNNBaseline
    print("✓ models.cnn_baseline.CNNBaseline")
except Exception as e:
    print(f"✗ models.cnn_baseline.CNNBaseline: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")
