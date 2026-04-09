#!/bin/bash
# Run all 9 SepFormer Posenc HPO validation experiments sequentially
# 3 configs × 3 seeds each
#
# Top-3 from sweep sepformer-posenc (0r4w3ep2):
#   Config 1: soft-sweep-6    (best_val_sisdr=6.077)
#   Config 2: happy-sweep-3   (best_val_sisdr=6.032)
#   Config 3: upbeat-sweep-8  (best_val_sisdr=6.018)

echo "Starting SepFormer Posenc HPO Validation (9 runs total)"
echo "======================================================="

SEEDS=(42 123 456)

for config_num in 1 2 3; do
    echo ""
    echo "Running Config $config_num..."

    config_file="experiments/sepformer/4-hpo-posenc-vals/validation_config${config_num}.yaml"

    for seed in "${SEEDS[@]}"; do
        echo "   Seed: $seed"
        python train.py --config "$config_file" --seed "$seed"
    done
done

echo ""
echo "All 9 validation runs complete!"
echo "Review results in WandB to compare configs across seeds."
