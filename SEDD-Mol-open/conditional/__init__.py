"""条件生成：给定 SMILES 前缀，让模型把剩下的部分补全。

条件生成本质上就是"无条件生成 + 前缀"：每一步去噪后把前缀和结束符写回序列，
模型只能补中间空出来的位置。不需要额外参数，也不需要微调。

    python conditional/cond_qm9.py   --model_path <run> --prefix "FOCC1=C"
    python conditional/cond_zinc.py  --model_path <run> --prefix "CC"
    python conditional/cond_moses.py --model_path <run> --prefix "CC"
"""
