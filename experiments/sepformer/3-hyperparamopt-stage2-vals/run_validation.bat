@echo off
REM Run all 9 SepFormer Stage 2 validation experiments sequentially
REM 3 configs x 3 seeds each
REM
REM Top-3 from sweep sepformer-stage2-8k (qqjh7cvm):
REM   Config 1: dutiful-sweep-9   (best_val_sisdr=4.300)
REM   Config 2: happy-sweep-19    (best_val_sisdr=4.096)
REM   Config 3: stoic-sweep-3     (best_val_sisdr=3.962)

echo Starting SepFormer Stage 2 Validation (9 runs total)
echo ==========================================================

for %%C in (1 2 3) do (
    echo.
    echo Running Config %%C...

    for %%S in (42 123 456) do (
        echo    Seed: %%S
        python train.py --config "experiments/sepformer/3-hyperparamopt-stage2-vals/validation_config%%C.yaml" --seed %%S
    )
)

echo.
echo All 9 validation runs complete!
echo Review results in WandB to compare configs across seeds.
