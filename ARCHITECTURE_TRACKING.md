# Architecture Tracking System

## Overview

The Architecture Tracking System in Neuro Exapt provides accurate tracking of neural network architecture changes during evolution, handling model wrapping and providing reliable visualization.

## Problem Solved

Previously, the visualization system incorrectly marked all layers as "new" in the final architecture, even when they were present from the beginning. This was caused by:

1. **Model Wrapping**: NeuroExaptWrapper adds a "model." prefix to all layer names
2. **Naive Comparison**: Simple layer name comparison failed due to prefix mismatch
3. **Lost History**: Evolution steps weren't properly tracked

## Solution

The new `ArchitectureTracker` class provides:

### 1. Wrapper-Aware Layer Tracking
```python
# Automatically detects wrapper prefixes
tracker = ArchitectureTracker()
tracker.initialize(wrapped_model, detect_wrapper=True)
# Detects "model." prefix and normalizes layer names
```

### 2. Evolution History
```python
# Track each evolution step
change = tracker.track_evolution(
    model, 
    epoch=10, 
    action='expand',
    details={'layers_added': 2}
)

# Query what changed
added = change.get_added_layers()      # ['conv6', 'conv7']
removed = change.get_removed_layers()   # []
modified = change.get_modified_layers() # ['fc1']
```

### 3. Accurate Layer Status
```python
# Get status relative to initial architecture
status = tracker.get_layer_status('conv1', 'initial')
# Returns: 'unchanged', 'new', 'removed', 'modified'
```

## Architecture Components

### LayerSnapshot
Captures layer state at a point in time:
- `name`: Normalized layer name
- `module_type`: Layer type (Conv2d, Linear, etc.)
- `parameters`: Parameter count
- `shape_info`: Input/output dimensions

### ArchitectureChange
Records architecture evolution:
- `epoch`: When change occurred
- `action`: Type of change (expand, prune, mutate)
- `affected_layers`: Layers that changed
- `layers_before/after`: Full snapshots for comparison

### ArchitectureTracker
Main tracking engine:
- Handles wrapper detection
- Normalizes layer names
- Tracks evolution history
- Provides accurate comparisons

## Integration

### In NeuroExapt
```python
class NeuroExapt:
    def __init__(self, ...):
        # ...
        self.architecture_tracker = ArchitectureTracker()
    
    def wrap_model(self, model):
        # ...
        self.architecture_tracker.initialize(self.wrapped_model)
```

### In Visualization
```python
visualizer.visualize_model(
    model,
    architecture_tracker=neuro_exapt.architecture_tracker
)
```

## Visual Output

### Before (Incorrect)
```
🏗️  Final Architecture
✓conv1 Conv   # Incorrectly marked as new
✓conv2 Conv   # Incorrectly marked as new
✓conv3 Conv   # Actually new
```

### After (Correct)
```
🏗️  Final Architecture
 conv1 Conv   # Correctly shown as unchanged
 conv2 Conv   # Correctly shown as unchanged
✓conv3 Conv   # Correctly marked as new
```

## Color Coding
- 🟢 Green (✓): New layers
- 🔴 Red (✗): Removed layers
- 🟡 Yellow (◇): Modified layers
- 🔵 Blue (space): Unchanged layers

## Usage Example

```python
from neuroexapt import NeuroExapt
from neuroexapt.utils.visualization import ModelVisualizer

# Initialize
neuro_exapt = NeuroExapt()
wrapped_model = neuro_exapt.wrap_model(your_model)

# Architecture is automatically tracked
# ...training and evolution...

# Visualize with accurate tracking
visualizer = ModelVisualizer()
visualizer.visualize_model(
    wrapped_model,
    title="Final Architecture",
    architecture_tracker=neuro_exapt.architecture_tracker
)
```

## Benefits

1. **Accurate Visualization**: Shows true architecture changes
2. **Evolution History**: Complete record of all changes
3. **Wrapper Handling**: Works with any model wrapper
4. **Layer Genealogy**: Track layer history through evolution
5. **Export Capability**: Save complete history for analysis

## API Reference

### ArchitectureTracker Methods

- `initialize(model, detect_wrapper=True)`: Initialize tracking
- `track_evolution(model, epoch, action, details)`: Record evolution
- `get_layer_status(layer_name, reference)`: Get layer status
- `get_evolution_summary()`: Get summary statistics
- `get_layer_history(layer_name)`: Get layer's complete history
- `export_history()`: Export full tracking data

### Integration Points

- NeuroExapt: Automatic tracking during evolution
- Visualization: Enhanced accuracy with tracker
- Analysis: Export data for detailed study 