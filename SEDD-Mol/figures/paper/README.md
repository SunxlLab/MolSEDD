# Camera-ready figure scripts

These scripts produce the final multi-panel figures of the paper. They are kept
verbatim from the submission repository, so they expect the original data layout
(log folders, generated molecule tables, SA/RAscore caches). Adjust the `DATA_*`
constants at the top of each script before running.

| script | figure |
|--------|--------|
| `redraw_nature_figures.py` | main multi-panel figures (distributions, t-SNE, logP panels) |
| `redraw_lzpneed_figures.py` | SA / RAscore panels |
| `redraw_ablation_7lines.py` | MCTS ablation, 7 curves |
| `redraw_cond_generated_logp_evolution.py` | logP evolution of conditionally generated molecules |
| `redraw_cond_xr_3methods.py` | conditional generation, 3 methods |
| `redraw_cond_xr_method_vs_dataset.py` | method × dataset comparison |
| `redraw_hit_rate_conditional_vs_unconditional.py` | hit rate, conditional vs. unconditional |
| `redraw_scaffold_snn_analysis.py` | scaffold / SNN analysis |
| `update_source_data_summary.py` | rebuilds `source_data_summary.csv` |

`figure_captions.txt` holds the captions, `source_data_summary.csv` the mapping from
each figure panel to its underlying data file.

Additional third-party dependencies: `seaborn`, `scikit-learn` (t-SNE), and the
SA/RAscore models from
[rdkit-contrib](https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score).
