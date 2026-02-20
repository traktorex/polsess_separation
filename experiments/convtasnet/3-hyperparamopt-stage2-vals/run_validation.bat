@echo off
REM Run all 9 ConvTasNet Stage 2 validation experiments sequentially
REM 3 configs x 3 seeds each
REM
REM Top-3 from sweep convtasnet-stage2-8k (71wtfegp):
REM   Config 1: stilted-sweep-16  (best_val_sisdr=3.358)
REM   Config 2: major-sweep-31    (best_val_sisdr=3.337)
REM   Config 3: quiet-sweep-8     (best_val_sisdr=3.332)

echo Starting ConvTasNet Stage 2 Validation (9 runs total)
echo ==========================================================

for %%C in (1 2 3) do (
    echo.
    echo Running Config %%C...

    for %%S in (42 123 456) do (
        echo    Seed: %%S
        python train.py --config "experiments/convtasnet/3-hyperparamopt-stage2-vals/validation_config%%C.yaml" --seed %%S
    )
)

echo.
echo All 9 validation runs complete!
echo Review results in WandB to compare configs across seeds.
