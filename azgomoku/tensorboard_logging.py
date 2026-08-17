"""Optional TensorBoard logging helpers for H3 and Arena runs."""

import json
from pathlib import Path


class NullWriter:
    """Keep training usable when the optional TensorBoard package is absent."""

    enabled = False

    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def create_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[TensorBoard] unavailable; install tensorboard to write event files", flush=True)
        return NullWriter()
    return SummaryWriter(log_dir=str(Path(log_dir)))


def log_config(writer, config):
    writer.add_text("config/run", json.dumps(config, sort_keys=True), 0)
    for key in ("board_size", "win_length", "seed", "hidden_dim", "mcts_playouts", "selfplay_games_per_iter", "learning_rate", "batch_size"):
        if key in config:
            writer.add_scalar(f"config/{key}", config[key], 0)


def log_device(writer, device):
    writer.add_text("performance/device", str(device), 0)
