#!/bin/bash
# Quick test script for baseline sweeps
# Tests each sweep with 1 run at 4 epochs

echo "Testing baseline sweep configurations..."
echo "========================================="
echo

cd /home/user/polsess_separation

# Test ConvTasNet
echo "1. Testing ConvTasNet sweep..."
echo "Creating temporary test config with 4 epochs..."

# Create test version with 4 epochs
sed 's/value: 100/value: 4/' sweeps/1-baselines-SB/convtasnet.yaml > sweeps/1-baselines-SB/convtasnet_test.yaml
sed -i 's/values: \[42, 123, 456\]/values: [42]/' sweeps/1-baselines-SB/convtasnet_test.yaml
sed -i 's/name: 1-baseline-SB-convtasnet/name: TEST-convtasnet/' sweeps/1-baselines-SB/convtasnet_test.yaml

echo "Run this command to test:"
echo "  wandb sweep sweeps/1-baselines-SB/convtasnet_test.yaml"
echo "  wandb agent <sweep_id>"
echo

# Cleanup
rm -f sweeps/1-baselines-SB/*_test.yaml

echo "========================================="
echo "Test Configuration Summary:"
echo "- 4 epochs (vs. 100)"
echo "- 1 seed (vs. 3)"
echo "- Same curriculum learning"
echo "- Same hyperparameters"
echo
echo "If test succeeds, run full sweeps with:"
echo "  wandb sweep sweeps/1-baselines-SB/convtasnet.yaml"
echo "  wandb agent <sweep_id> --count 3"
