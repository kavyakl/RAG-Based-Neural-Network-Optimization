# Dataset Crosscheck Report: yeast

Generated on: 2025-04-15 09:33:42

## Data Summary

- Results Index: 6 entries
- Activation Comparison: 3 entries
- Profiling Data: 3 files
- Original Metrics: 6 files
- Pruned Metrics: 0 files

## Activation Function Comparison

| Source | Activation Functions |
|--------|---------------------|
| Results Index | sigmoid, tanh, relu |
| Activation Comparison | sigmoid, tanh, relu |

Activation functions match between Results Index and Activation Comparison

## Accuracy Comparison

| Activation | Results Index Accuracy | Activation Comparison Accuracy | Difference |
|------------|------------------------|------------------------------|------------|
| sigmoid | 41.08% | 37.37% | 3.70% |
| tanh | 55.05% | 54.55% | 0.51% |
| relu | 48.48% | 49.49% | 1.01% |

## Pruning Rate Comparison

| Activation | Results Index Pruning Rate | Activation Comparison Pruning Rate | Difference |
|------------|----------------------------|----------------------------------|------------|
| sigmoid | 59.38% | 59.38% | 0.00% |
| tanh | 59.38% | 59.38% | 0.00% |
| relu | 59.38% | 59.38% | 0.00% |

## Profiling Data Summary

| Activation | Device | Inference Time (ms) | ROM Size (bytes) | RAM Usage (bytes) |
|------------|--------|---------------------|------------------|-------------------|
| relu (original) | raspberry-pi-4 | 1.00 | 38880 | 3448 |
| relu (original) | raspberry-pi-rp2040 | 10.00 | 38880 | 3448 |
| relu (pruned) | raspberry-pi-4 | 1.00 | 38880 | 3448 |
| relu (pruned) | raspberry-pi-rp2040 | 7.00 | 38880 | 3448 |
| tanh (original) | raspberry-pi-4 | 1.00 | 38880 | 3448 |
| tanh (original) | raspberry-pi-rp2040 | 10.00 | 38880 | 3448 |
| tanh (pruned) | raspberry-pi-4 | 1.00 | 38880 | 3448 |
| tanh (pruned) | raspberry-pi-rp2040 | 7.00 | 38880 | 3448 |
| sigmoid (original) | raspberry-pi-4 | 1.00 | 38880 | 3448 |
| sigmoid (original) | raspberry-pi-rp2040 | 13.00 | 38880 | 3448 |
| sigmoid (pruned) | raspberry-pi-4 | 1.00 | 38880 | 3448 |
| sigmoid (pruned) | raspberry-pi-rp2040 | 7.00 | 38880 | 3448 |

