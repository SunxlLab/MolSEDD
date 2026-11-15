"""MCTS 性质定向生成。

搜索逐 token 生长 SMILES 前缀，用 SEDD 自己的下一 token 分布做 PUCT 先验，
朝目标性质（Crippen logP）推进。

    search.py    搜索引擎（选择 / 扩展 / 回传 / reward）
    run_*.py     论文主方法，每个数据集一个
"""
