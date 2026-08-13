from dataclasses import asdict, dataclass

@dataclass
class Config:
    board_size: int = 6
    win_length: int = 4
    mcts_playouts: int = 50
    self_play_games: int = 30
    replay_capacity: int = 5000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    attention_heads: int = 4
    train_epochs: int = 2
    training_iterations: int = 1
    c_puct: float = 1.5
    temperature: float = 1.0
    opening_temperature_moves: int = 10
    late_temperature: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    symmetry_augmentation: bool = True
    seed: int = 7
    @classmethod
    def profile(cls, name):
        cfg=cls()
        if name=="smoke": cfg.mcts_playouts,cfg.self_play_games=10,2
        elif name=="pilot": cfg.mcts_playouts,cfg.self_play_games=100,100
        elif name!="default": raise ValueError(name)
        return cfg
    def dict(self): return asdict(self)
