#!/bin/bash
# Run all 9 ConvTasNet Stage 2 validation experiments sequentially
# 3 configs × 3 seeds each
#
# Top-3 from sweep convtasnet-stage2-8k (71wtfegp):
#   Config 1: stilted-sweep-16  (best_val_sisdr=3.358)
#   Config 2: major-sweep-31    (best_val_sisdr=3.337)
#   Config 3: quiet-sweep-8     (best_val_sisdr=3.332)

echo "🚀 Starting ConvTasNet Stage 2 Validation (9 runs total)"
echo "=========================================================="

# Seeds to test
SEEDS=(42 123 456)

# Run each config with all 3 seeds
for config_num in 1 2 3; do
    echo ""
    echo "📊 Running Config $config_num..."

    config_file="experiments/convtasnet/3-hyperparamopt-stage2-vals/validation_config${config_num}.yaml"

    for seed in "${SEEDS[@]}"; do
        echo "   Seed: $seed"

        # Run training with --seed argument
        python train.py --config "$config_file" --seed "$seed"
    done
done

echo ""
echo "✅ All 9 validation runs complete!"
echo "Review results in WandB to compare configs across seeds."
