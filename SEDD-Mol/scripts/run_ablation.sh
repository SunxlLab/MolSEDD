#!/usr/bin/env bash
# Run the MCTS variants of the ablation figure sequentially on one GPU.
# Each run writes a log plus a trajectory CSV into $OUT.
#
#   MODEL_PATH=exp_local/moses/2026.02.26/104218 ./scripts/run_ablation.sh
#   VARIANTS="puct uct uct_novalid pure" ./scripts/run_ablation.sh   # four-line figure only
#
# Every variable below can be overridden from the environment.

set -u

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a training run directory}
DATASET=${DATASET:-moses}
TARGET=${TARGET:-1.0}
SIMULATIONS=${SIMULATIONS:-500}
C_PARAM=${C_PARAM:-0.618}
DISCOVERY_BS=${DISCOVERY_BS:-200}
SEED=${SEED:-42}
OUT=${OUT:-condgen_out/ablation}
PYTHON=${PYTHON:-python}
VARIANTS=${VARIANTS:-"puct uct uct_novalid pure puct_uniform puct_novalid puct_uniform_novalid"}

mkdir -p "$OUT"

for variant in $VARIANTS; do
  echo "=== $variant ==="
  "$PYTHON" -m condgen.run_mcts \
      --variant "$variant" \
      --dataset "$DATASET" \
      --model_path "$MODEL_PATH" \
      --wanti "$TARGET" \
      --simulations "$SIMULATIONS" \
      --c_param "$C_PARAM" \
      --discovery_bs "$DISCOVERY_BS" \
      --seed "$SEED" \
      --output_dir "$OUT" \
      2>&1 | tee "$OUT/${DATASET}_${variant}.log"
done

echo "Done. Trajectories are in $OUT"
echo "Plot with:  python figures/plot_ablation.py --run \"ours=$OUT/${DATASET}_puct_*.csv\" ..."
