#!/usr/bin/env bash
# 顺序跑多个消融变体，每个变体一份 log。
#
#   MODEL_PATH=exp_local/moses/2026.02.26/104218 ./ablation/run_ablation.sh
#
# 可覆盖的环境变量：
#   MODEL_PATH     训练产出的实验目录（必填）
#   DATASET        moses | qm9 | zinc            （默认 moses）
#   TARGET         目标 logP                     （默认 1.0）
#   VARIANTS       逗号分隔的变体名              （默认 all，即论文四条线）
#   SIMULATIONS    覆盖该数据集的搜索次数        （默认用脚本内默认值）
#   C_PARAM        覆盖 c                        （默认用脚本内默认值）
#   SEED           随机种子                      （默认 42）
#   OUT            输出目录                      （默认 mcts_out/ablation）
#   PYTHON         python 解释器                 （默认 python）
#
# 四个变体：puct（本文方法）/ uct / uct_novalid / pure

set -euo pipefail

DATASET=${DATASET:-moses}
TARGET=${TARGET:-1.0}
VARIANTS=${VARIANTS:-all}
SEED=${SEED:-42}
OUT=${OUT:-mcts_out/ablation}
PYTHON=${PYTHON:-python}

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "请先设置 MODEL_PATH，例如：" >&2
  echo "  MODEL_PATH=exp_local/moses/2026.02.26/104218 $0" >&2
  exit 1
fi

mkdir -p "$OUT"

EXTRA=()
[[ -n "${SIMULATIONS:-}" ]] && EXTRA+=(--simulations "$SIMULATIONS")
[[ -n "${C_PARAM:-}" ]] && EXTRA+=(--c_param "$C_PARAM")

"$PYTHON" ablation/run_ablation.py \
  --dataset "$DATASET" \
  --model_path "$MODEL_PATH" \
  --variants "$VARIANTS" \
  --wanti "$TARGET" \
  --seed "$SEED" \
  --output_dir "$OUT" \
  "${EXTRA[@]}" 2>&1 | tee "$OUT/${DATASET}_ablation.log"
