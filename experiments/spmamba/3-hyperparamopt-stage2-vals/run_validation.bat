@echo off
REM Run all 9 SPMamba Stage 2 validation experiments sequentially
REM 3 configs x 3 seeds each
REM
REM Top-3 from sweep spmamba-stage2-8k (2wdpj22v):
REM   Config 1: glowing-sweep-9   (best_val_sisdr=5.264)
REM   Config 2: autumn-sweep-10   (best_val_sisdr=5.120)
REM   Config 3: cerulean-sweep-2  (best_val_sisdr=5.103)

echo Starting SPMamba Stage 2 Validation (9 runs total)
echo =======================================================

for %%C in (1 2 3) do (
    echo.
    echo Running Config %%C...

    for %%S in (42 123 456) do (
        echo    Seed: %%S
        python train.py --config "experiments/spmamba/3-hyperparamopt-stage2-vals/validation_config%%C.yaml" --seed %%S
    )
)

echo.
echo All 9 validation runs complete!
echo Review results in WandB to compare configs across seeds.
