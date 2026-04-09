#!/bin/bash
# Run all 6 MambaTasNet-XS HPO validation experiments sequentially
# 3 configs × 3 seeds each
#
# Top-3 from sweep mamba-tasnet-xs (gbyurf31):
#   Config 1: rosy-sweep-7     (best_val_sisdr=4.126)
#   Config 2: valiant-sweep-23 (best_val_sisdr=4.062)
#   Config 3: rural-sweep-5    (best_val_sisdr=4.047)

echo "Starting MambaTasNet-XS HPO Validation (9 runs total)"
echo "======================================================="

SEEDS=(42 123 456)

for config_num in 1 2 3; do
    echo ""
    echo "Running Config $config_num..."

    config_file="experiments/mamba_tasnet/5-hpo-xs-vals/validation_config${config_num}.yaml"

    for seed in "${SEEDS[@]}"; do
        echo "   Seed: $seed"
        python train.py --config "$config_file" --seed "$seed"
    done
done

echo ""
echo "All 6 validation runs complete!"
echo "Review results in WandB to compare configs across seeds."
