# Run all 9 SepFormer Posenc HPO validation experiments sequentially
# 3 configs x 3 seeds each
#
# Top-3 from sweep sepformer-posenc (0r4w3ep2):
#   Config 1: soft-sweep-6    (best_val_sisdr=6.077)
#   Config 2: happy-sweep-3   (best_val_sisdr=6.032)
#   Config 3: upbeat-sweep-8  (best_val_sisdr=6.018)

Write-Host "Starting SepFormer Posenc HPO Validation (9 runs total)"
Write-Host "======================================================="

$seeds = @(42, 123, 456)

foreach ($configNum in 1..3) {
    Write-Host ""
    Write-Host "Running Config $configNum..."

    $configFile = "experiments/sepformer/4-hpo-posenc-vals/validation_config${configNum}.yaml"

    foreach ($seed in $seeds) {
        Write-Host "   Seed: $seed"
        python train.py --config $configFile --seed $seed
    }
}

Write-Host ""
Write-Host "All 9 validation runs complete!"
Write-Host "Review results in WandB to compare configs across seeds."
