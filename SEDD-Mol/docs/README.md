# Documentation

| file | language | content |
|------|----------|---------|
| `mcts_formulas.md` | EN/中文 | Reward definition `R = exp(-MAE)`, UCT vs. PUCT selection, candidate discovery and normalisation, dead-node handling, worked numerical example |
| `four_line_ablation_zh.md` | 中文 | Definition of the four curves of the ablation figure (method → switches mapping) |
| `ablation_notes_zh.md` | 中文 | Full ablation log written during the experiments (per-run observations) |
| `project_overview_zh.md` | 中文 | Project overview: datasets, pipeline, experiment plan |
| `hyperparameters_zh.md` | 中文 | Complete hyper-parameter table (model, training, optimizer, sampling, MCTS) |
| `speed_benchmark_zh.md` | 中文 | Sampling speed measurements and their interpretation |

The Chinese documents are the original lab notes; they use the old script names.
For their mapping to this repository see section 11 of the top-level `README.md`:

| old name | new entry point |
|---|---|
| `MOSES_1_our.py` | `condgen/run_mcts.py --variant puct` |
| `MOSEScond_GGTMCTS.py` | `condgen/run_mcts.py --variant uct` |
| `MOSES_5_UCT无有效性检测.py` | `condgen/run_mcts.py --variant uct_novalid` |
| `MOSEScond_PureMCTS.py` | `condgen/run_mcts.py --variant pure` |
| `MOSES_2_减去频率.py` | `condgen/run_mcts.py --variant puct_uniform` |
| `MOSES_3_减去有效性检测.py` | `condgen/run_mcts.py --variant puct_novalid` |
| `MOSES_4_减去有效性检测和频率.py` | `condgen/run_mcts.py --variant puct_uniform_novalid` |
