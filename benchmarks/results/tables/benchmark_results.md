# CartPole Benchmark Results

**Date**: 2025-10-25T20:05:04.549168

**Environment**: CartPole-v1

**Model**: models/cartpole_hybrid_real_model.pth

## Standard Adaptation

| Config | Adapt Steps | LR | Avg Reward | Std | Min | Max |
|--------|-------------|-----|------------|-----|-----|-----|
| 1 | 5 | 0.01 | 9.10 | 0.77 | 8.0 | 10.0 |
| 2 | 10 | 0.01 | 8.85 | 0.79 | 8.0 | 10.0 |
| 3 | 10 | 0.001 | 13.50 | 1.80 | 10.0 | 17.0 |

## Hybrid Adaptation

| Config | Adapt Steps | LR | Avg Reward | Std | Min | Max |
|--------|-------------|-----|------------|-----|-----|-----|
| 1 | 5 | 0.01 | 37.75 | 17.07 | 20.0 | 83.0 |
| 2 | 10 | 0.01 | 9.55 | 0.67 | 8.0 | 11.0 |
| 3 | 10 | 0.001 | 10.70 | 3.62 | 9.0 | 24.0 |

## Comparison

| Config | Standard | Hybrid | Improvement |
|--------|----------|--------|-------------|
| 1 | 9.10 | 37.75 | +314.8% |
| 2 | 8.85 | 9.55 | +7.9% |
| 3 | 13.50 | 10.70 | -20.7% |
