#!/bin/bash
# Run all 9 Experiment B validation experiments sequentially
# 3 configs × 3 seeds each

echo "🚀 Starting Experiment B Final Validation (9 runs total)"
echo "=========================================================="

# Seeds to test
SEEDS=(42 123 456)

# Run each config with all 3 seeds
for config_num in 1 2 3; do
    echo ""
    echo "📊 Running Config $config_num..."
    
    config_file="experiments/dprnn/3-hyperparamopt-expB-vals/validation_config${config_num}.yaml"
    
    for seed in "${SEEDS[@]}"; do
        echo "   Seed: $seed"
        
        # Run training with --seed argument
        python train.py --config "$config_file" --seed "$seed"
    done
done

echo ""
echo "✅ All 9 validation runs complete!"
echo "Review results in WandB to compare with 3-stage approach."
