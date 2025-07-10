# Neuro Exapt Examples

This directory contains examples demonstrating how to use the Neuro Exapt framework for various tasks.

## Basic Examples

### 1. Basic Classification (`basic_classification.py`)

A simple example showing how to use Neuro Exapt for image classification on CIFAR-10:

```bash
cd examples
python basic_classification.py
```

**Features demonstrated:**
- Model wrapping with NeuroExapt
- Information-theoretic loss integration
- Dynamic architecture evolution
- Training progress monitoring
- Visualization of evolution history

**Expected behavior:**
- The model will automatically prune redundant layers when entropy drops
- New layers may be added when information bottlenecks are detected
- Training metrics and structure evolution will be logged

## Running Examples

### Prerequisites

Make sure you have Neuro Exapt installed:

```bash
pip install -e .
```

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- At least 4GB RAM

### Data Download

Most examples will automatically download required datasets on first run. Make sure you have sufficient disk space:

- CIFAR-10: ~170 MB
- MNIST: ~10 MB

## Advanced Usage

Each example includes detailed comments explaining:

1. **Model Initialization**: How to create and configure your base model
2. **NeuroExapt Setup**: Configuration options and parameters
3. **Training Loop**: Integration with your existing training code
4. **Monitoring**: How to track information-theoretic metrics
5. **Analysis**: Post-training model analysis and visualization

## Customization

The examples can be easily modified for your own use cases:

- Replace the model architecture with your own
- Use your own datasets by modifying the data loading code
- Adjust hyperparameters for different behavior
- Add custom information-theoretic metrics

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Slow convergence**: Adjust learning rate or evolution frequency
3. **No evolution events**: Check entropy thresholds and information weights

### Performance Tips

1. Use mixed precision training for faster computation
2. Enable cudnn benchmark for consistent input sizes
3. Use data loading with multiple workers
4. Monitor GPU utilization during training

## Contributing

To add new examples:

1. Create a new Python file in this directory
2. Follow the existing code structure and comments
3. Include a brief description in this README
4. Test with different configurations

## Support

For questions about the examples:
- Check the main documentation
- Open an issue on GitHub
- Contact the development team 