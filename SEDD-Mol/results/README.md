# Results

Search logs and metric tables behind the figures of the paper. They are kept in
version control (they are small) so the figures can be regenerated without
re-running the (GPU-hours long) searches.

## `logs/`

| file | corresponds to |
|------|----------------|
| `moses_puct_sedd_prior.txt` | proposed method (PUCT + SEDD prior, dynamic vocabulary, validity filter), target logP = 1 |
| `moses_uct_dynamic_vocab.txt` | UCT + dynamic vocabulary |
| `moses_pure_mcts.txt` | pure MCTS (static vocabulary) |
| `moses_puct_sedd_prior_run.log` | `run_ablation.sh` run #1 (`MOSES_1_our.py`) |
| `moses_puct_uniform_run.log` | ablation, uniform prior (`MOSES_2_减去频率.py`) |
| `moses_puct_novalidity_run.log` | ablation, no validity filter (`MOSES_3_减去有效性检测.py`) |
| `moses_puct_uniform_novalidity_run.log` | ablation, neither (`MOSES_4_减去有效性检测和频率.py`) |
| `moses_uct_dynamic_novalidity_run.log` | ablation, UCT without validity filter (`MOSES_5_UCT无有效性检测.py`) |
| `moses_puct_best_prefixes.txt` | best prefixes discovered by the proposed method |

Log lines look like

```
[Disc & Sim] Prefix: CC | Eff: 91.00% | AvgLogP: 0.9812 | MAE: 0.0188 | R: 0.9814 | ...
```

`figures/plot_ablation.py` reads the newer `condgen_out/*.csv` trajectories; to use
these historical logs instead, parse the `R:` values in the order they appear and
take the running maximum (best-so-far).

## `moses_5seed_results_table.md`

Mean ± std over 5 random seeds for the four methods of the ablation figure.
