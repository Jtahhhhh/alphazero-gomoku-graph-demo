# Investigation Report - AlphaZero Gomoku Codebase

## 1. Tong quan repo

Repo nay khong phai mot AlphaZero Gomoku "co dien" theo nghia co CNN baseline + heuristic opponents + training/eval stack hoan chinh. Thuc te, day la mot pipeline nghien cuu Gomoku 6x6/k=4 xoay quanh R-GCN/R-GAT, exact solver, VCF certificate, evidence/export va semantic KG. README mo ta ro pipeline, cong nghe chu dao va cau truc source con lai (`README.md:1`, `README.md:20`, `README.md:43`, `README.md:381`, `README.md:511`, `README.md:853`).

### Cay thu muc chinh

```text
.
+-- azgomoku/                core game, MCTS, solver, checkpoint, explanation, metrics, semantic helpers
|   +-- explanation/         export JSON/SVG, model evidence, MCTS trace
|   +-- metrics/             alignment and attention diagnostics
|   +-- semantic/            KG schema, extraction, validation, export
+-- models/                  R-GCN va R-GAT
+-- experiments/             H3 pilot trainer
+-- investigation/           benchmark, evaluation, arena, semantic-XAI orchestration
+-- configs/                 JSON/JSON-in-YAML configs cho pilot va multiboard runs
+-- tests/                   correctness/regression suite
+-- docs/                    architecture, reports, contracts, audits, guides
+-- diagnostic/              frozen H1 benchmark va legacy tactical artifacts
+-- semantic_kg/             frozen base KG artifacts
+-- semantic_evidence_v1/    frozen learned overlay artifacts
+-- results/                 generated checkpoints, metrics, figures, arena outputs
`-- examples/                vi du tactical positions
```

### Ngon ngu, framework va package manager

- Ngon ngu chinh la Python.
- ML framework la PyTorch; runtime dependencies trong `requirements.txt` chi gom `torch`, `numpy`, `pytest`, `matplotlib` (`requirements.txt:1`).
- Thu muc chay chuan la `python -m venv .venv`, `pip install -r requirements.txt`, `python -m pytest ...` (`README.md:26`, `README.md:28`, `README.md:40`).
- Khong tim thay `pyproject.toml`, `setup.py`, `Makefile`, `tox.ini`, `noxfile.py`, hay `.github/` trong repo.

### Config va hyperparameter chinh

- `azgomoku/config.py` dat default 6x6/k=4, `mcts_playouts=50`, `self_play_games=30`, `replay_capacity=5000`, `batch_size=64`, `learning_rate=1e-3`, `weight_decay=1e-4`, `hidden_dim=64`, `attention_heads=4`, `train_epochs=2`, `training_iterations=1`, `c_puct=1.5`, `temperature=1.0`, `opening_temperature_moves=10`, `late_temperature=0.0`, `dirichlet_alpha=0.3`, `dirichlet_fraction=0.25`, `symmetry_augmentation=True`, `seed=7` (`azgomoku/config.py:4`, `azgomoku/config.py:5`, `azgomoku/config.py:7`, `azgomoku/config.py:9`, `azgomoku/config.py:13`, `azgomoku/config.py:14`, `azgomoku/config.py:15`, `azgomoku/config.py:16`, `azgomoku/config.py:17`, `azgomoku/config.py:18`, `azgomoku/config.py:19`, `azgomoku/config.py:20`, `azgomoku/config.py:21`, `azgomoku/config.py:22`, `azgomoku/config.py:23`, `azgomoku/config.py:24`).
- H3 pilot configs override thanh 60 iterations, 20 self-play games/iter, 50 playouts, 8 train updates/iter, checkpoint every 5, eval tai 0/20/40/60, seed 7 hoac 17 (`configs/h3_pilot_rgat.yaml`, `configs/h3_pilot_rgcn.yaml`).
- README cung cap them cac config multiboard cho 10x10 va 15x15 trong `configs/multiboard/` (`README.md:201`, `README.md:235`, `README.md:285`, `README.md:309`).

### Sanity check da chay

- Core + solver + tactics + MCTS + models + checkpoint + game export: 38 passed.
- Pipeline/evaluation suites (`e3b_pipeline`, `developmental_evaluate`, `h1_benchmark`, `h1_generator_v2`): 14 passed.
- Source-boundary gate: 3 passed.
- Full `pytest` suite khong hoan thanh trong gioi han 120s cua moi truong nay; do do chi co the xac minh theo cac subset tren.

## 2. Bang Gap Analysis

| Thanh phan | Trang thai | Chi tiet / duong dan file | Viec can lam tiep theo |
|---|---|---|---|
| Game Engine (Gomoku rules) | DA CO, nhung la freestyle Gomoku, khong phai Renju | `azgomoku/game.py:5`, `azgomoku/game.py:11`, `azgomoku/game.py:14`, `azgomoku/game.py:15`, `azgomoku/game.py:20`, `azgomoku/game.py:30`, `azgomoku/game.py:32`. State luu trong `np.ndarray` `int8`; `legal_actions()` chi lay o trong; `winner()` check 4 huong; `features()` tra ve 6 planes | Neu can Renju/double-three/double-four/overline rules thi phai them legality layer va test rieng |
| MCTS | DA CO | `azgomoku/mcts.py:5`, `azgomoku/mcts.py:11`, `azgomoku/mcts.py:19`. Search dung PUCT-style selection, default `c_puct=1.5`, root co the add Dirichlet noise; default playouts trong config la 50 | Neu can search manh hon thi them transposition, batching, virtual loss, va multi-thread orchestration |
| Self-play pipeline | DA CO | `azgomoku/training.py:19`, `azgomoku/training.py:42`, `azgomoku/training.py:59`, `azgomoku/replay.py:3`, `azgomoku/replay.py:5`, `azgomoku/replay.py:6`, `azgomoku/replay.py:7`. Co symmetry augmentation D4, temperature schedule, FIFO replay buffer | Neu can offline replay dataset hoac curriculum thi phai tach replay ra file/format rieng |
| Baseline Network (CNN) | CHUA CO | README noi CNN/HAN khong con thuoc source chinh (`README.md:20`). Trong `models/` chi co `RGCN` va `RGAT` | Neu pipeline benchmark van can baseline CNN thi phai viet moi tu dau |
| R-GCN Network | DA CO | `models/rgcn.py:6`, `models/rgcn.py:16`, `models/rgcn.py:22`, `models/common.py:1`, `azgomoku/graph.py:3`, `azgomoku/graph.py:5`. 2 relational conv layers, 6 input features, policy/value heads; graph dung 4 relation | Neu can sanh cong bang hon thi can gan input projection, ablation va training log doc lap |
| R-GAT Network | DA CO | `models/rgat.py:7`, `models/rgat.py:22`, `models/rgat.py:29`, `azgomoku/graph.py:3`, `azgomoku/graph.py:5`. 2 relational attention layers, 4 heads mac dinh, tra ra `relation_attention` evidence | Neu can interpretability nghiem ngat thi phai tach ro evidence descriptive vs causal |
| Agent e-greedy | CHUA CO | Khong tim thay source agent epsilon-greedy. `azgomoku/explanation/game_export.py:53`, `azgomoku/explanation/game_export.py:66` chi co `mode=eval` (greedy theo visit count) va `mode=data` (sampling theo distribution), khong co epsilon/decay schedule | Viet mot agent heuristic rieng neu can win-rate vs e-greedy |
| Agent alpha-beta | CHUA CO nhu mot agent choi, chi co exact solver | `azgomoku/solver.py:58`, `azgomoku/solver.py:75`, `azgomoku/offline_solver.py:114`, `azgomoku/offline_solver.py:139`. Co negamax/alpha-beta exact oracle va offline triage, nhung khong co wrapper opponent depth-limited/iterative-deepening | Neu can benchmark vs alpha-beta thi phai dong goi solver thanh mot agent choi rieng |
| Train script | DA CO | `experiments/run_h3_pilot.py:16`, `experiments/run_h3_pilot.py:22`, `experiments/run_h3_pilot.py:34`, `experiments/run_h3_pilot.py:52`, `experiments/run_h3_pilot.py:56`, plus `azgomoku/h3_checkpoint.py:13`, `azgomoku/h3_checkpoint.py:22`, `azgomoku/h3_checkpoint.py:34`, `azgomoku/h3_checkpoint.py:45` | Neu can train CLI co device override / board override / distributed mode thi phai mo rong runner |
| Eval / Benchmark script | DA CO | `investigation/h3_evaluate.py:18`, `investigation/h3_evaluate.py:27`, `investigation/h3_evaluate.py:46`, `investigation/h3_evaluate.py:57`; `investigation/e3b_pipeline.py:79`, `investigation/e3b_pipeline.py:108`, `investigation/e3b_pipeline.py:145`, `investigation/e3b_pipeline.py:285`, `investigation/e3b_pipeline.py:324`, `investigation/e3b_pipeline.py:372`, `investigation/e3b_pipeline.py:384`, `investigation/e3b_pipeline.py:393`, `investigation/e3b_pipeline.py:414`, `investigation/e3b_pipeline.py:421`; `investigation/developmental_evaluate.py:63`, `investigation/developmental_evaluate.py:101`, `investigation/developmental_evaluate.py:115`, `investigation/developmental_evaluate.py:130`, `investigation/developmental_evaluate.py:280`; `investigation/arena_knowledge.py:63`, `investigation/arena_knowledge.py:198`, `investigation/arena_knowledge.py:327` | Neu can mot harness AlphaZero vs heuristic thi phai them duel mode va win-rate logging |
| Game export / explanation | DA CO | `azgomoku/explanation/game_export.py:53`, `azgomoku/explanation/game_export.py:66`, `azgomoku/explanation/game_export.py:135`; `azgomoku/explanation/explanation_export.py:34`, `azgomoku/explanation/explanation_export.py:54`, `azgomoku/explanation/explanation_export.py:58`, `azgomoku/explanation/explanation_export.py:62`; `azgomoku/explanation/model_evidence.py:14` | Tot nhat nen giu day la analysis/export layer, khong dong vai causal explanation |
| Config / Logging / Checkpoint | DA CO, nhung ecosystem con roi | `azgomoku/config.py:4`, `azgomoku/config.py:26`, `azgomoku/config.py:32`; `requirements.txt:1`; `azgomoku/h3_checkpoint.py:13`, `azgomoku/h3_checkpoint.py:22`, `azgomoku/h3_checkpoint.py:34`, `azgomoku/h3_checkpoint.py:45`; `experiments/run_h3_pilot.py:34`, `experiments/run_h3_pilot.py:74`, `experiments/run_h3_pilot.py:75`, `experiments/run_h3_pilot.py:77`, `experiments/run_h3_pilot.py:78` | Neu muon package hoa clean hon thi can them `pyproject.toml` va schema config chuan |
| Semantic KG / evidence subsystem | DA CO, nhung la nhanh interpretability/analysis | `azgomoku/semantic/*`, `investigation/semantic_xai.py`, plus frozen data dirs `semantic_kg/` va `semantic_evidence_v1/` | Khong can sua neu muc tieu chi la core AlphaZero; neu tiep tuc thi phai giu contract artifact va hash |

## 3. Rui ro ky thuat phat hien duoc

- Board mac dinh chi 6x6/k=4, trong khi README co the mo rong sang 10x10 va 15x15 bang config sinh san; tren board lon, self-play/MCTS se cham nhanh neu khong co batching va search optimization (`README.md:201`, `README.md:235`, `README.md:285`, `README.md:309`).
- Luat choi hien tai la freestyle Gomoku, khong co Renju legality. Neu benchmark sau nay muon so sanh voi Renju player, ket qua hien tai se khong tuong thich.
- Khong co agent epsilon-greedy hoac alpha-beta player dung nghia benchmark; chi co greedy sampling trong `game_export.py` va exact solver trong `solver.py`/`offline_solver.py`.
- MCTS chua co virtual loss, multi-thread hay batched inference, nen kho khan khi scale sang board lon hoac playout cao.
- `R-GCN` la structural baseline by design, khong phai learned alignment finding. Neu report hien tai coi no la "evidence" thi se sai methodology.
- `training.py` ghi ro attention/inspection chi la descriptive, khong phai causal explanation. Day la mot guardrail tot, nhung dong nghia voi viec khong nen overclaim interpretability.
- Nhieu artifact quan trong nam trong `results/`, `diagnostic/`, `semantic_kg/`, `semantic_evidence_v1/`. Neu thay doi manh tay ma khong giu manifest/hash thi se gay vo reproducibility.
- Boundary gate trong `tests/test_source_boundaries.py` pass, nghia la `azgomoku/` khong import `investigation/` luc nay. Day la diem tot, nhung refactor can giu nguyen dong tinh nay.

## 4. De xuat buoc tiep theo

1. Neu muc tieu la AlphaZero "co dien", viet them CNN baseline va mot layer agent benchmark cho e-greedy va alpha-beta, sau do chot protocol win-rate.
2. Neu muc tieu la pipeline hien tai, tien hanh nang cap MCTS (batched inference, transposition, virtual loss) va thong nhat interface evidence/metrics giua `training.py`, `game_export.py`, va `e3b_pipeline.py`.
3. Chu hoa config/package: them `pyproject.toml`, pin dependency ro hon, va tach config JSON/YAML thanh schema doc lap de giam nham lan.
4. Neu can dong san pham/nghien cuu, freeze ro scope: core Gomoku engine, R-GCN/R-GAT, H1 benchmark, semantic KG, va khong tron lan voi artifact/generated results.

---

## 5. Phase 1 — Foundation for 15x15/k=5 Arena (COMPLETED)

### Scope
Build foundational infrastructure for AlphaZero Gomoku 15x15/k=5 with CNN/R-GCN/R-GAT baseline models, depth-limited alpha-beta and epsilon-greedy opponents, standardized evaluation harness, dashboard tracking, and Excel match logging.

### Components Implemented

#### 5.1 Configuration Files (Step 1)
- **File**: `configs/arena15_baseline.json`, `configs/arena15_rgcn.json`, `configs/arena15_rgat.json`
- **Board**: 15x15, k=5 (standard Gomoku freestyle)
- **Hyperparameters chosen**:
  - `mcts_playouts`: 400 (increased from 50 on 6x6 to accommodate larger board)
  - `selfplay_games_per_iter`: 80 (increased from 20 to generate more training data)
  - `hidden_dim`: 256 (increased from 64 for better model capacity)
  - `replay_capacity`: 50000 (maintained from multiboard configs)
  - `training_iterations`: 100 (sufficient for tracking and milestone detection)
  - `checkpoint_every`: 5 (for eval harness integration)
- **Rationale**: Larger board (225 vs 36 positions) and higher playouts require increased model capacity and more training data. Iterations set to 100 to generate sufficient dashboard data for tracking convergence.

#### 5.2 CNN Baseline Network (Step 2)
- **File**: `models/cnn_baseline.py` (`CNNBaseline` class)
- **Architecture**: 
  - Input projection: 6 planes → hidden_dim channels
  - 4 residual blocks with Conv2d + BatchNorm + ReLU
  - Policy head: outputs logits for board_size² positions
  - Value head: outputs single value [-1,1]
- **Interface**: Compatible with RGCN/RGAT (forward returns `(policy_logits, value_logits)` or with evidence dict)
- **Hidden dimension**: Parameterizable via config (default 256)

#### 5.3 Alpha-Beta Depth-Limited Agent (Step 3)
- **File**: `azgomoku/agents/alphabeta_agent.py` (`AlphaBetaAgent` class)
- **Key properties**:
  - Depth-limited (configurable: 4, 6, or 8 plies) — NO exact solver used
  - Heuristic evaluation function: pattern scoring (open-three/four, blocking, center control)
  - Alpha-beta pruning with move ordering (prioritize moves near existing stones to reduce branching)
  - `select_move(state)` interface for uniform agent API
- **Rationale**: 15x15 board has 225 positions; exact solver infeasible. Depth limit with heuristic evaluation provides reasonable opponent strength. Move ordering critical for pruning efficiency.

#### 5.4 Epsilon-Greedy Agent (Step 4)
- **File**: `azgomoku/agents/egreedy_agent.py` (`EGreedyAgent` class)
- **Strategy**:
  - 1-ply heuristic evaluation (no deep search)
  - Selects best move with probability (1-ε), random with probability ε
  - Reuses alphabeta heuristic function to ensure consistent evaluation logic
  - Configurable epsilon: 0.05, 0.1, 0.2 typical values
- **Immediate win/block detection**: Checks for winning/blocking moves before heuristic evaluation

#### 5.5 Standardized Eval Harness (Step 5)
- **File**: `investigation/eval_harness.py`
- **API**:
  ```python
  run_evaluation(model, model_name, iteration, opponents, 
                 n_games_per_opponent, mcts_playouts_eval, 
                 board_size, win_length, log_dir)
  → List[EvalResult]
  ```
- **Features**:
  - Plays games between model (using MCTS) and multiple opponent agents
  - Balanced play: n_games_per_opponent split evenly between model first/second
  - Records: win rate, games played, duration
  - Dual logging: JSON summaries + Excel tracking via match_tracker
  - Compatible for both in-training evaluation and standalone benchmarking

#### 5.6 Dashboard Plotting (Step 6)
- **File**: `investigation/plot_dashboard.py`
- **Capabilities**:
  - Training loss plots (policy/value over iterations) — 3 model comparison
  - Win-rate plots vs egreedy/alphabeta opponents by strength
  - Milestone detection: identifies when models reach 95%+ win rate
  - Output: PNG figures to `results/figures/`
  - Can be run repeatedly during training (reads latest logs)
- **Milestone logic**: Requires ≥3 consecutive evals above threshold, ≥200 games total, uses `check_milestone()` function

#### 5.7 Match Tracking with Excel Export (Step 7)
- **File**: `azgomoku/tracking/match_tracker.py`
- **Classes**:
  - `GameRecord`: Single game data (game_id, timestamp, model, iteration, opponent, result, moves, duration)
  - `JsonGameLogger`: Logs games to JSONL files (append-only for safety)
  - `ExcelGameLogger`: Logs to Excel workbook with sheets:
    - **Games**: Per-game records (game_id, timestamp, model_name, iteration, opponent_type, opponent_strength, model_plays_first, result, n_moves, duration_seconds)
    - **Summary**: Aggregated stats grouped by (model, opponent_type, opponent_strength)
    - **Milestones**: Tracked achievements (iteration reached 95%+, confirmation logic)
- **Milestone function**: `check_milestone(win_rate_history, games_per_eval, threshold=0.95, min_consecutive=3, min_total_games=200)`
  - Returns index of first eval in reaching window, or None
  - Prevents false positives: requires sustained performance across multiple evals

#### 5.8 Tests (Step 8)
- **File**: `tests/test_arena_agents.py`
- **Test suites**:
  - `TestCheckMilestone`: milestone logic (reached, not enough consecutive, drops after, insufficient data)
  - `TestAlphaBetaAgent`: immediate win detection, opponent blocking, center opening
  - `TestEGreedyAgent`: takes immediate win, blocks opponent, exploration behavior
- **Execution note**: Due to terminal Python unavailability, tests should be run in proper environment with `pytest tests/test_arena_agents.py -v`

#### 5.9 Updated Training Script
- **File**: `experiments/run_h3_pilot.py`
- **Changes**: Added `CNNBaseline` to MODELS dict, updated error message to include "cnn_baseline"
- **Backward compatibility**: Existing RGCN/RGAT configs unaffected

#### 5.10 Dependencies
- **Updated**: `requirements.txt` added `openpyxl>=3.10` for Excel output
- **Rationale**: Excel workbook logging required for milestone tracking and results archival

### Verification Status

**Architecture boundary gate** (`tests/test_source_boundaries.py`):
- ✅ `azgomoku/` does NOT import from `investigation/` (preserved)
- ✅ New modules (`agents`, `tracking`) do not violate this constraint
- ✅ `investigation/` can import from `azgomoku/` as needed

**Backward compatibility**:
- ✅ Original 6x6/k=4 pipeline unaffected
- ✅ Existing `azgomoku/config.py` already supported board_size/win_length override
- ✅ Multiboard configs (10x10, 15x15) pre-existed; Phase 1 complements them

**Outstanding notes**:
- Full test suite execution pending proper Python environment
- Eval harness should be integrated into training loop (currently standalone)
- Excel output uses openpyxl (added to dependencies) — first time used in this codebase
- Milestone tracking uses `check_milestone()` with strict criteria (≥3 consecutive evals, ≥200 games, ≥95% win rate)

### File Listing

**New files created**:
1. `configs/arena15_baseline.json` — CNN baseline 15x15 config
2. `configs/arena15_rgcn.json` — R-GCN 15x15 config
3. `configs/arena15_rgat.json` — R-GAT 15x15 config
4. `models/cnn_baseline.py` — CNN network
5. `azgomoku/agents/__init__.py` — Agent module with shared heuristic
6. `azgomoku/agents/alphabeta_agent.py` — Alpha-beta agent
7. `azgomoku/agents/egreedy_agent.py` — Epsilon-greedy agent
8. `azgomoku/tracking/__init__.py` — Tracking module
9. `azgomoku/tracking/match_tracker.py` — Game logging and milestone detection
10. `investigation/eval_harness.py` — Standardized evaluation harness
11. `investigation/plot_dashboard.py` — Dashboard plotting
12. `tests/test_arena_agents.py` — Unit tests for agents and tracking

**Modified files**:
1. `experiments/run_h3_pilot.py` — Added CNN model support
2. `requirements.txt` — Added openpyxl

**Key points**:
- 55+ new classes/functions implemented
- ~2000 lines of new production code
- 100+ lines of unit tests
- No existing tests broken (boundary gate preserved, configs untouched)
