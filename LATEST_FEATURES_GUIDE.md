# NeuroExapt Latest Features Guide

## Overview

The examples in this repository have been updated to showcase all the latest features of NeuroExapt. This guide explains how to use these features effectively.

## 🚀 Quick Start

Run the basic classification example:
```bash
python examples/basic_classification.py
```

For deep learning targeting 95%+ accuracy:
```bash
python examples/deep_classification.py
```

To test features without full training:
```bash
python test_latest_features.py
```

## 📦 Key Features

### 1. Automatic Batch Size Optimization with Caching

The system automatically finds the optimal batch size for your GPU and caches it:

```python
# First run - optimizes batch size
python examples/basic_classification.py
# Output: "Finding optimal batch size..."
# Output: "✅ Optimal batch size: 928"
# Output: "💾 Cached optimal batch size for future use"

# Second run - uses cached value
python examples/basic_classification.py  
# Output: "📦 Using cached optimal batch size: 928"
# Output: "   GPU: NVIDIA GeForce RTX 3060 Laptop GPU"
# Output: "   ⚠️  If you changed GPU, delete cache at: ~/.neuroexapt/cache"
```

To skip batch size optimization:
```bash
# Windows
set SKIP_BATCH_OPTIMIZATION=true
python examples/basic_classification.py

# Linux/Mac
SKIP_BATCH_OPTIMIZATION=true python examples/basic_classification.py
```

To clear the cache:
```python
from neuroexapt.utils.gpu_manager import gpu_manager
gpu_manager.clear_batch_size_cache()
```

### 2. Colorful Architecture Visualization

The system now shows architecture changes with colors in the terminal:

- 🟢 **Green**: New layers added during evolution
- 🔴 **Red + Strikethrough**: Removed layers
- 🟡 **Yellow**: Modified layers (parameter changes)
- 🔵 **Blue**: Unchanged layers
- 🔷 **Cyan**: Shape annotations
- 🔸 **Gray**: Parameter counts

Example output:
```
🏗️  Architecture after expand (Epoch 15)
================================================================================

📊 Sequential Architecture
-----------------------------------------

        Input: [3,32,32]
              ↓
         conv1 Conv (9.4K)
          [64,H,W]
              ↓
         conv2 Conv (36.9K)
          [128,H,W]
              ↓
    ✓conv3 Conv (147.6K)   ← NEW LAYER!
          [256,H,W]
              ↓
         fc1 FC (262.4K)
          [1024]
              ↓
         fc2 FC (10.2K)
          [10]

================================================================================
📊 Total Parameters: 466.5K
📈 Parameter Change: +147.6K
```

### 3. Entropy-Based Evolution

Evolution now triggers based on entropy saturation, not fixed epochs:

```
Epoch 12 | Train: 92.45% | Val: 91.23% | Entropy: 0.287/0.250 | Time: 8.2s

🔄 Entropy-based evolution triggered!
   Entropy saturated at high level (variance < 0.01 over 10 epochs)
✅ Evolution completed: expand
   Parameters: 319,178 → 466,794 (+147,616)
```

The system monitors:
- Entropy saturation (low variance)
- Accuracy plateaus
- Overfitting gaps
- Performance stagnation

### 4. Advanced Progress Tracking

Real-time progress with detailed metrics:

```
Epoch 25/50
Training: 67% | Loss: 0.4521 | Acc: 89.34%

Epoch 25 | Train Loss: 0.4512 | Train Acc: 89.42% | Val Acc: 88.76% | 
Entropy: 0.312/0.280 | Time: 7.9s

📊 GPU Stats:
   Memory: 2.1/6.0 GB
   Utilization: 98%
   Temperature: 72°C
```

### 5. GPU Optimization Features

- **Mixed Precision Training**: Automatic FP16 training for faster computation
- **Non-blocking Transfers**: Overlapped CPU-GPU communication [[memory:2921760]]
- **cuDNN Optimization**: Enabled benchmark mode for convolutions
- **Memory Management**: Automatic cache clearing and optimization

### 6. Advanced Dataset Loading

The examples use AdvancedDatasetLoader with:
- P2P acceleration for faster downloads
- 迅雷 (Thunder) integration for Chinese users
- Automatic caching and resume support
- Integrity verification

## 🎯 Usage Tips

### For Best Performance

1. **Let batch size optimization run once** - it caches the result
2. **Use default settings** - they're optimized for your hardware
3. **Monitor GPU stats** - shown every 10 epochs
4. **Watch entropy values** - evolution triggers automatically

### For Experimentation

1. **Force evolution**: The system will force expansion if accuracy is too low
2. **Adjust thresholds**: Modify `entropy_threshold` for different evolution rates
3. **Change models**: Switch between 'simple' and 'deeper' models

### For Deep Learning (95%+ Accuracy)

Run the deep classification example:
```bash
python examples/deep_classification.py
```

Features:
- More aggressive evolution
- Longer training (100 epochs)
- Target accuracy tracking
- Early stopping when target reached

## 🔧 Troubleshooting

### Batch Size Issues

If you see warnings about GPU memory:
```python
# Use more conservative settings
batch_size = gpu_manager.get_optimal_batch_size(
    model, 
    input_shape=(3, 32, 32),
    safety_factor=0.8  # Use only 80% of memory
)
```

### Cache Problems

Clear all caches:
```bash
rm -rf ~/.neuroexapt/cache  # Linux/Mac
rmdir /s %USERPROFILE%\.neuroexapt\cache  # Windows
```

### Evolution Not Triggering

Check entropy values - evolution requires:
- Entropy variance < 0.01 (saturation)
- Or accuracy plateau for multiple epochs
- Or manual override when accuracy is low

## 📊 Example Outputs

### Successful Training
```
🎉 Training Completed!
================================================================================

📊 Final Results:
   Best validation accuracy: 93.45%
   Total training time: 42.3 minutes
   Initial parameters: 319,178
   Final parameters: 614,410
   Parameter increase: 295,232 (92.5%)
   Final entropy: 0.298

🔄 Evolution Summary:
   Total evolution events: 3
   1. Epoch 15: expand (+147,616 params, acc: 87.23%, entropy: 0.342)
   2. Epoch 28: expand (+147,616 params, acc: 90.12%, entropy: 0.315)
   3. Epoch 41: prune (-24,384 params, acc: 92.89%, entropy: 0.195)
```

### Architecture Evolution
```
🏗️  Final Architecture:
   Base layers: 7
   Expansion layers: 2 (conv6, conv7)
   Total layers: 9
   Architecture: Sequential → Expanded → Optimized
```

## 🚀 Next Steps

1. **Try different models**: Modify the model architectures in the examples
2. **Adjust hyperparameters**: Experiment with learning rates, entropy thresholds
3. **Monitor evolution**: Watch how the architecture adapts to your data
4. **Use in your projects**: Apply NeuroExapt to your own models and datasets

Happy experimenting with NeuroExapt! 🎉 