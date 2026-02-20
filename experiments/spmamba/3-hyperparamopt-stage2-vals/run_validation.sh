#!/bin/bash
# Run all 9 SPMamba Stage 2 validation experiments sequentially
# 3 configs × 3 seeds each
#
# Top-3 from sweep spmamba-stage2-8k (2wdpj22v):
#   Config 1: glowing-sweep-9   (best_val_sisdr=5.264)
#   Config 2: autumn-sweep-10   (best_val_sisdr=5.120)
#   Config 3: cerulean-sweep-2  (best_val_sisdr=5.103)

echo "🚀 Starting SPMamba Stage 2 Validation (9 runs total)"
echo "======================================================="

# Seeds to test
SEEDS=(42 123 456)

# Run each config with all 3 seeds
for config_num in 1 2 3; do
    echo ""
    echo "📊 Running Config $config_num..."

    config_file="experiments/spmamba/3-hyperparamopt-stage2-vals/validation_config${config_num}.yaml"

    for seed in "${SEEDS[@]}"; do
        echo "   Seed: $seed"

        # Run training with --seed argument
        python train.py --config "$config_file" --seed "$seed"
    done
done

echo ""
echo "✅ All 9 validation runs complete!"
echo "Review results in WandB to compare configs across seeds."
