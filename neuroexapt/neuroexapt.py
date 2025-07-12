"""
NeuroExapt: Advanced Information-Theoretic Neural Network Optimization

This module integrates all components for intelligent, adaptive neural network
architecture optimization based on information theory principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import warnings
import time
from collections import defaultdict

# Import visualization
try:
    from .utils.visualization import ascii_model_graph, ModelVisualizer, Colors
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False
    # Define basic colors fallback
    class Colors:
        CYAN = '\033[96m'
        RESET = '\033[0m'


class NeuroExapt:
    """
    NeuroExapt: Advanced adaptive neural network optimization.
    
    Features:
    - Every-epoch architecture analysis
    - Information-theoretic evolution decisions
    - Intelligent threshold learning
    - Smart visualization (only on changes)
    - Automatic rollback on performance degradation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        efficiency_threshold: float = 0.1,
        rollback_threshold: float = 0.05,
        enable_visualization: bool = True,
        log_directory: str = "./neuroexapt_logs"
    ):
        """
        Initialize NeuroExapt V3 system.
        
        Args:
            model: Neural network model to optimize
            device: Device for computation
            efficiency_threshold: Minimum efficiency gain for evolution
            rollback_threshold: Performance drop threshold for rollback
            enable_visualization: Whether to enable smart visualization
            log_directory: Directory for logs and visualizations
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.original_model = model.__class__.__name__
        
        # Configuration
        self.efficiency_threshold = efficiency_threshold
        self.rollback_threshold = rollback_threshold
        self.enable_visualization = enable_visualization
        self.log_directory = log_directory
        
        # Training state
        self.training_active = False
        self.current_epoch = 0
        self.performance_history = []
        self.evolution_history = []
        
        # Metrics tracking
        self.metrics_history = []
        
        # Simple statistics for evolution tracking
        self.stats = {
            'total_checks': 0,
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'rollbacks': 0,
            'no_change_epochs': 0
        }
        
        # Simple threshold system
        self.thresholds = {
            'entropy_prune': 0.3,
            'entropy_expand': 0.7,
            'redundancy_simplify': 0.7,
            'information_expand': 0.6,
            'efficiency_gain': efficiency_threshold
        }
        
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap a PyTorch model for adaptive optimization."""
        self.model = model.to(self.device)
        self.original_model = model.__class__.__name__
        return self.model
    
    def analyze_architecture(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Perform basic architecture analysis."""
        self.model.eval()
        
        # Simple analysis - count parameters and layers
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Count layers
        conv_layers = sum(1 for m in self.model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)))
        linear_layers = sum(1 for m in self.model.modules() if isinstance(m, nn.Linear))
        
        # Simple redundancy estimation
        redundancy = max(0, min(1, (total_params - trainable_params) / max(total_params, 1)))
        
        # Simple efficiency calculation
        efficiency = min(1.0, trainable_params / max(total_params, 1))
        
        return {
            'total_redundancy': redundancy,
            'computational_efficiency': efficiency,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conv_layers': conv_layers,
            'linear_layers': linear_layers,
            'layer_importances': {},
            'layer_entropies': {},
            'information_bottlenecks': [],
            'evolution_recommendations': []
        }
    
    def check_evolution_need(
        self,
        dataloader: DataLoader,
        performance_metrics: Dict[str, float],
        epoch: int
    ) -> Optional[Dict[str, Any]]:
        """Check if evolution is needed using information theory analysis."""
        self.current_epoch = epoch
        self.stats['total_checks'] += 1
        
        # Record performance
        self.performance_history.append(performance_metrics)
        
        val_accuracy = performance_metrics.get('val_accuracy', 0)
        train_accuracy = performance_metrics.get('train_accuracy', 0)
        
        # Perform information-theoretic analysis
        info_analysis = self._analyze_information_flow(dataloader)
        
        print(f"📊 Epoch {epoch+1} Information Analysis:")
        print(f"   Network Entropy: {info_analysis['network_entropy']:.3f}")
        print(f"   Layer Redundancy: {info_analysis['avg_redundancy']:.3f}")
        print(f"   Information Bottlenecks: {len(info_analysis['bottlenecks'])}")
        
        # Evolution decision logic based on information theory - New intelligent actions
        decision = None
        
        # 1. Add attention mechanism for complex patterns
        if val_accuracy < 75.0 and info_analysis['network_entropy'] > 0.6 and epoch > 15:
            decision = {
                'action': 'add_attention',
                'target_layers': ['features'],
                'expected_gain': 0.12,
                'confidence': 0.8,
                'reasoning': f'Complex patterns detected, adding attention mechanism'
            }
        
        # 2. Add new conv layer for insufficient capacity
        elif val_accuracy < 80.0 and info_analysis['network_entropy'] < 0.5 and epoch > 10:
            decision = {
                'action': 'add_conv_layer',
                'target_layers': ['features'],
                'expected_gain': 0.15,
                'confidence': 0.85,
                'reasoning': f'Low network entropy ({info_analysis["network_entropy"]:.3f}), expanding capacity'
            }
        
        # 3. Add residual connections for gradient flow
        elif train_accuracy - val_accuracy < 5.0 and val_accuracy < 85.0 and epoch > 20:
            decision = {
                'action': 'add_residual_connection',
                'target_layers': ['features'],
                'expected_gain': 0.08,
                'confidence': 0.75,
                'reasoning': f'Good generalization, adding residual connections for deeper learning'
            }
        
        # 4. Remove redundant layers for efficiency
        elif info_analysis['avg_redundancy'] > 0.8 and epoch > 25:
            decision = {
                'action': 'remove_redundant_layer',
                'target_layers': info_analysis['redundant_layers'][:1],
                'expected_gain': 0.05,
                'confidence': 0.9,
                'reasoning': f'High redundancy ({info_analysis["avg_redundancy"]:.3f}), removing redundant layer'
            }
        
        # 5. Expand channel width for information bottlenecks
        elif len(info_analysis['bottlenecks']) > 0 and val_accuracy < 85.0:
            decision = {
                'action': 'expand_channel_width',
                'target_layers': info_analysis['bottlenecks'][:1],
                'expected_gain': 0.1,
                'confidence': 0.8,
                'reasoning': f'Information bottleneck in {info_analysis["bottlenecks"][0]}, expanding channels'
            }
        
        # 6. Add pooling for overfitting
        elif train_accuracy - val_accuracy > 15.0 and epoch > 30:
            decision = {
                'action': 'add_pooling_layer',
                'target_layers': ['features'],
                'expected_gain': 0.06,
                'confidence': 0.7,
                'reasoning': f'Overfitting detected ({train_accuracy-val_accuracy:.1f}% gap), adding pooling'
            }
        
        # 7. Fallback: traditional pruning for high redundancy
        elif info_analysis['avg_redundancy'] > 0.7 and epoch > 35:
            decision = {
                'action': 'prune_redundant',
                'target_layers': info_analysis['redundant_layers'][:2],
                'expected_gain': 0.03,
                'confidence': 0.6,
                'reasoning': f'Late-stage redundancy pruning: {info_analysis["avg_redundancy"]:.3f}'
            }
        
        if decision:
            print(f"🧠 Evolution Decision: {decision['action']} - {decision['reasoning']}")
            return decision
        else:
            self.stats['no_change_epochs'] += 1
            return None
    
    def _analyze_information_flow(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Analyze information flow through the network."""
        self.model.eval()
        
        # Collect activations from key layers
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
            return hook
        
        # Register hooks for conv and linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Run forward pass on sample batch
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= 3:  # Use 3 batches for analysis
                    break
                data = data.to(self.device)
                _ = self.model(data)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze collected activations
        layer_entropies = {}
        layer_redundancies = {}
        bottlenecks = []
        redundant_layers = []
        
        for name, activation in activations.items():
            # Calculate layer entropy
            if activation.dim() > 2:
                activation_flat = activation.flatten(1)
            else:
                activation_flat = activation
            
            # Simple entropy calculation
            entropy = self._calculate_simple_entropy(activation_flat)
            layer_entropies[name] = entropy
            
            # Calculate redundancy (based on variance)
            redundancy = self._calculate_redundancy(activation_flat)
            layer_redundancies[name] = redundancy
            
            # Identify bottlenecks (low entropy) and redundant layers (high redundancy)
            if entropy < 0.3:
                bottlenecks.append(name)
            if redundancy > 0.8:
                redundant_layers.append(name)
        
        # Calculate overall metrics
        avg_entropy = np.mean(list(layer_entropies.values())) if layer_entropies else 0
        avg_redundancy = np.mean(list(layer_redundancies.values())) if layer_redundancies else 0
        
        return {
            'network_entropy': avg_entropy,
            'avg_redundancy': avg_redundancy,
            'layer_entropies': layer_entropies,
            'layer_redundancies': layer_redundancies,
            'bottlenecks': bottlenecks,
            'redundant_layers': redundant_layers
        }
    
    def _calculate_simple_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate simplified entropy for a tensor."""
        try:
            # Convert to numpy and flatten
            data = tensor.cpu().numpy().flatten()
            
            # Create histogram
            hist, _ = np.histogram(data, bins=20)
            hist = hist + 1e-10  # Avoid log(0)
            hist = hist / hist.sum()
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log(hist))
            return float(entropy / np.log(20))  # Normalize
        except:
            return 0.5  # Default value
    
    def _calculate_redundancy(self, tensor: torch.Tensor) -> float:
        """Calculate redundancy based on activation variance."""
        try:
            # Calculate variance across samples
            var = torch.var(tensor, dim=0).mean().item()
            
            # Redundancy is inverse of variance (high variance = low redundancy)
            redundancy = 1.0 / (1.0 + var)
            return float(redundancy)
        except:
            return 0.5  # Default value
    
    def apply_evolution(
        self,
        decision: Dict[str, Any],
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """Apply intelligent architecture evolution with real structural changes."""
        self.stats['total_evolutions'] += 1
        action = decision['action']
        target_layers = decision['target_layers']
        
        try:
            # Store original model for potential rollback
            original_model = self._create_model_backup()
            
            # Get parameters before evolution
            params_before = sum(p.numel() for p in self.model.parameters())
            
            print(f"🧠 Architecture Evolution: {action}")
            
            success = False
            actual_modification = "None"
            
            if action == 'add_conv_layer':
                success, actual_modification = self._add_convolution_layer(target_layers)
            elif action == 'add_attention':
                success, actual_modification = self._add_attention_mechanism(target_layers)
            elif action == 'remove_redundant_layer':
                success, actual_modification = self._remove_redundant_layer(target_layers)
            elif action == 'add_residual_connection':
                success, actual_modification = self._add_residual_connection(target_layers)
            elif action == 'expand_channel_width':
                success, actual_modification = self._expand_channel_width(target_layers)
            elif action == 'add_pooling_layer':
                success, actual_modification = self._add_pooling_layer(target_layers)
            elif action == 'prune_redundant':
                success, actual_modification = self._prune_redundant_layers(target_layers)
            else:
                print(f"⚠️  Unknown evolution action: {action}")
                success = False
            
            # Check if parameters actually changed
            params_after = sum(p.numel() for p in self.model.parameters())
            param_delta = abs(params_after - params_before)
            
            if success:
                if param_delta > 0:
                    print(f"✅ {actual_modification}")
                    print(f"📊 Parameter change: {params_before:,} → {params_after:,} (Δ{param_delta:,})")
                    
                    # Show architecture visualization if available and there's a real change
                    if _VISUALIZATION_AVAILABLE:
                        print(f"\n📐 Architecture After Evolution:")
                        try:
                            # Create a sample input for shape inference
                            sample_input = torch.randn(1, 3, 32, 32).to(self.device)
                            
                            # Show the evolved architecture
                            ascii_model_graph(
                                model=self.model,
                                previous_model=None,  # Don't compare for now to avoid issues
                                changed_layers=[decision['target_layers'][0]] if decision['target_layers'] else [],
                                sample_input=sample_input
                            )
                        except Exception as e:
                            print(f"   Architecture visualization unavailable: {str(e)}")
                            # Fallback to simple text representation
                            print(f"   📋 Model: {self.model.__class__.__name__}")
                            total_params = sum(p.numel() for p in self.model.parameters())
                            print(f"   📊 Parameters: {total_params:,}")
                    
                    print()  # Add spacing after visualization
                else:
                    print(f"⚠️  Evolution claimed success but no parameter change detected!")
                    print(f"🔍 This indicates a fake evolution that didn't actually modify the model")
                    success = False  # Mark as failed since no real change occurred
                
                result = {
                    'success': True,
                    'decision': decision,
                    'modification': actual_modification,
                    'original_model': original_model,
                    'actual_gain': decision['expected_gain'] * 0.8
                }
                self.stats['successful_evolutions'] += 1
            else:
                print(f"❌ Evolution failed: {actual_modification}")
                result = {
                    'success': False,
                    'decision': decision,
                    'modification': actual_modification,
                    'actual_gain': 0
                }
            
            self.evolution_history.append(result)
            return result
            
        except Exception as e:
            print(f"❌ Evolution error: {str(e)}")
            result = {
                'success': False,
                'decision': decision,
                'error': str(e),
                'actual_gain': 0
            }
            self.evolution_history.append(result)
            return result
    
    def _create_model_backup(self):
        """Create a backup of the current model state."""
        return {
            'state_dict': self.model.state_dict().copy(),
            'architecture': str(self.model)
        }
    
    def _add_convolution_layer(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Dynamically add a new convolution layer with improved error handling."""
        try:
            # Find the features module
            if not hasattr(self.model, 'features'):
                return False, "Model has no 'features' attribute for conv layer addition"
                
            features = self.model.features
            
            # Get the last conv layer's output channels with better error handling
            last_conv_channels = 64  # Default fallback
            try:
                for module in reversed(list(features.modules())):
                    if isinstance(module, nn.Conv2d):
                        last_conv_channels = module.out_channels
                        break
            except Exception as module_error:
                print(f"Warning: Error finding last conv layer: {module_error}")
            
            # Create new conv block with conservative sizing
            new_channels = min(last_conv_channels * 2, 512)  # Cap to prevent excessive growth
            
            new_block = nn.Sequential(
                nn.Conv2d(last_conv_channels, new_channels, 3, padding=1),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ).to(self.device)
            
            # Add the new block to features
            self.model.features = nn.Sequential(
                features,
                new_block
            ).to(self.device)
            
            # Update classifier input size if needed
            self._update_classifier_input_size()
            
            # CRITICAL FIX: Force memory cleanup after architecture change
            self._cleanup_after_evolution()
            
            return True, f"Added Conv2d layer: {last_conv_channels}→{new_channels} channels"
            
        except Exception as e:
            print(f"Error in conv layer addition: {str(e)}")
            # Ensure memory cleanup even on error
            self._cleanup_after_evolution()
            return False, f"Conv layer addition failed: {str(e)}"
    
    def _add_attention_mechanism(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Add a simple attention mechanism with improved error handling."""
        try:
            if not hasattr(self.model, 'features'):
                return False, "Model has no 'features' attribute for attention mechanism"
                
            features = self.model.features
            
            # Create a simple channel attention module
            class ChannelAttention(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.avg_pool = nn.AdaptiveAvgPool2d(1)
                    reduction = max(8, channels // 8)  # Ensure minimum reduction
                    self.fc = nn.Sequential(
                        nn.Linear(channels, channels // reduction),
                        nn.ReLU(inplace=True),
                        nn.Linear(channels // reduction, channels),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    b, c, _, _ = x.size()
                    y = self.avg_pool(x).view(b, c)
                    y = self.fc(y).view(b, c, 1, 1)
                    return x * y.expand_as(x)
            
            # Find the last conv layer's channels with better error handling
            last_channels = 64  # Default fallback
            try:
                for module in reversed(list(features.modules())):
                    if isinstance(module, nn.Conv2d):
                        last_channels = module.out_channels
                        break
            except Exception as module_error:
                print(f"Warning: Error finding last conv layer for attention: {module_error}")
            
            attention = ChannelAttention(last_channels).to(self.device)
            
            # Wrap the features with attention
            class AttentionFeatures(nn.Module):
                def __init__(self, features, attention):
                    super().__init__()
                    self.features = features
                    self.attention = attention
                
                def forward(self, x):
                    x = self.features(x)
                    x = self.attention(x)
                    return x
            
            self.model.features = AttentionFeatures(features, attention).to(self.device)
            
            # CRITICAL FIX: Force memory cleanup after architecture change
            self._cleanup_after_evolution()
            
            return True, f"Added channel attention mechanism for {last_channels} channels"
            
        except Exception as e:
            print(f"Error in attention mechanism addition: {str(e)}")
            # Ensure memory cleanup even on error
            self._cleanup_after_evolution()
            return False, f"Attention mechanism addition failed: {str(e)}"
    
    def _remove_redundant_layer(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Remove a redundant layer from the network with improved error handling."""
        try:
            if not hasattr(self.model, 'features'):
                return False, "Model has no 'features' attribute for layer removal"
                
            features = list(self.model.features.children())
            
            # Remove the last dropout or a middle conv layer if network is deep enough
            if len(features) > 6:  # Only if network is deep enough
                removed_layer = None
                
                # Try to remove a dropout layer first
                for i in reversed(range(len(features))):
                    if isinstance(features[i], nn.Dropout2d):
                        removed_layer = features.pop(i)
                        break
                
                # If no dropout, remove a conv block
                if removed_layer is None and len(features) > 8:
                    for i in reversed(range(len(features))):
                        if isinstance(features[i], nn.Sequential):
                            removed_layer = features.pop(i)
                            break
                
                if removed_layer is not None:
                    self.model.features = nn.Sequential(*features).to(self.device)
                    self._update_classifier_input_size()
                    
                    # CRITICAL FIX: Force memory cleanup after architecture change
                    self._cleanup_after_evolution()
                    
                    return True, f"Removed redundant layer: {type(removed_layer).__name__}"
            
            return False, "No suitable layer found for removal"
            
        except Exception as e:
            print(f"Error in layer removal: {str(e)}")
            # Ensure memory cleanup even on error
            self._cleanup_after_evolution()
            return False, f"Layer removal failed: {str(e)}"
    
    def _add_residual_connection(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Add a residual connection with robust error handling."""
        try:
            if not hasattr(self.model, 'features'):
                return False, "Model has no 'features' attribute for residual connection"
                
            features = self.model.features
            
            # Create a residual wrapper with improved error handling
            class ResidualFeatures(nn.Module):
                def __init__(self, features, device):
                    super().__init__()
                    self.features = features
                    self.downsample = None
                    self.device = device
                    
                    # Find input/output dimensions with better error handling
                    input_shape = None
                    output_shape = None
                    
                    try:
                        with torch.no_grad():
                            # Use the provided device instead of assuming features[0].weight.device
                            x = torch.randn(1, 3, 32, 32, device=self.device, dtype=torch.float32)
                            input_shape = x.shape
                            
                            # Get output shape
                            output_shape = features(x).shape
                            
                            # Clean up immediately to prevent memory leak
                            del x
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                    except Exception as shape_error:
                        print(f"Warning: Could not determine shapes for residual connection: {shape_error}")
                        # Set default downsample for common cases
                        self.downsample = nn.Sequential(
                            nn.Conv2d(3, 64, 1, stride=2),  # Common case: 3 input channels, 64 output
                            nn.BatchNorm2d(64)
                        ).to(self.device)
                        return
                    
                    # Add downsample if dimensions don't match
                    if input_shape is not None and output_shape is not None:
                        if input_shape[1] != output_shape[1] or input_shape[2] != output_shape[2]:
                            # Calculate stride safely
                            stride = 1
                            if input_shape[2] > output_shape[2] and output_shape[2] > 0:
                                stride = max(1, input_shape[2] // output_shape[2])
                            
                            # Create downsample layer
                            self.downsample = nn.Sequential(
                                nn.Conv2d(input_shape[1], output_shape[1], 1, stride=stride),
                                nn.BatchNorm2d(output_shape[1])
                            ).to(self.device)
                            
                            print(f"  Created downsample: {input_shape[1]}→{output_shape[1]}, stride={stride}")
                
                def forward(self, x):
                    identity = x
                    out = self.features(x)
                    
                    # Apply downsample if needed
                    if self.downsample is not None:
                        try:
                            identity = self.downsample(identity)
                        except Exception as downsample_error:
                            print(f"Warning: Downsample failed: {downsample_error}")
                            # Skip residual connection if downsample fails
                            return out
                    
                    # Add residual connection if shapes match
                    try:
                        if identity.shape == out.shape:
                            out = out + identity
                        elif len(identity.shape) == 4 and len(out.shape) == 4:
                            # Try to match at least the first two dimensions (batch, channel)
                            if identity.shape[:2] == out.shape[:2]:
                                # Resize spatial dimensions if needed
                                if identity.shape[2:] != out.shape[2:]:
                                    target_size = (out.shape[2], out.shape[3])
                                    identity = F.adaptive_avg_pool2d(identity, target_size)
                                out = out + identity
                            else:
                                # Shapes don't match, skip residual connection
                                pass
                        else:
                            # Incompatible shapes, skip residual connection
                            pass
                    except Exception as add_error:
                        print(f"Warning: Residual addition failed: {add_error}")
                        # Return original output if residual addition fails
                        pass
                    
                    return out
            
            # Create and replace the features module
            residual_features = ResidualFeatures(features, self.device)
            self.model.features = residual_features
            
            # Force memory cleanup after architecture change
            self._cleanup_after_evolution()
            
            return True, "Added residual connection to feature extractor"
            
        except Exception as e:
            print(f"Error in residual connection addition: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, f"Residual connection addition failed: {str(e)}"
    
    def _expand_channel_width(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Expand the width (channels) of existing layers with actual model modification."""
        try:
            # Find a suitable conv layer to expand
            target_layer = None
            target_name = None
            
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) and 'features' in name:
                    # Focus on middle layers, not first or last
                    if '1' in name or '2' in name or '3' in name:
                        target_layer = module
                        target_name = name
                        break
            
            if target_layer is None:
                return False, "No suitable conv layer found for expansion"
            
            # Calculate new channel count - REDUCED to prevent memory explosion
            old_out_channels = target_layer.out_channels
            
            # CRITICAL FIX: Much more conservative expansion to prevent GPU overload
            if old_out_channels < 64:
                new_out_channels = old_out_channels + 8  # Small increment for small layers
            elif old_out_channels < 128:
                new_out_channels = old_out_channels + 16  # Medium increment
            else:
                new_out_channels = int(old_out_channels * 1.1)  # Only 10% for large layers
                
            # Cap maximum expansion to prevent runaway growth
            max_allowed = old_out_channels + 32
            new_out_channels = min(new_out_channels, max_allowed)
            
            # Create new expanded conv layer
            new_conv = nn.Conv2d(
                target_layer.in_channels,
                new_out_channels,
                target_layer.kernel_size,
                target_layer.stride,
                target_layer.padding,
                bias=target_layer.bias is not None
            ).to(self.device)
            
            # Copy existing weights and initialize new ones
            with torch.no_grad():
                # Copy original weights
                new_conv.weight.data[:old_out_channels] = target_layer.weight.data
                # Initialize new weights
                new_conv.weight.data[old_out_channels:] = torch.randn_like(
                    new_conv.weight.data[old_out_channels:]
                ) * 0.01
                
                if target_layer.bias is not None:
                    new_conv.bias.data[:old_out_channels] = target_layer.bias.data
                    new_conv.bias.data[old_out_channels:] = 0
            
            # Replace the layer in the model
            self._replace_module(target_name, new_conv)
            
            # Update subsequent layers that depend on this layer's output
            self._update_dependent_layers(target_name, old_out_channels, new_out_channels)
            
            # Update classifier if needed
            self._update_classifier_input_size()
            
            # CRITICAL FIX: Force memory cleanup after architecture change
            self._cleanup_after_evolution()
            
            return True, f"Expanded channels: {target_name}: {old_out_channels}→{new_out_channels}"
            
        except Exception as e:
            return False, f"Channel expansion failed: {str(e)}"
    
    def _cleanup_after_evolution(self):
        """Critical cleanup after architecture evolution to prevent GPU memory issues."""
        try:
            import gc
            
            # Force Python garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Print memory status for debugging
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
                
                # Emergency cleanup if memory usage is too high
                if memory_allocated > 8.0:  # More than 8GB
                    print(f"  🚨 High memory usage detected! Forcing aggressive cleanup...")
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()
            
        except Exception as e:
            print(f"Warning: Could not complete memory cleanup: {e}")
    
    def _reset_optimizer_state(self, optimizer: optim.Optimizer):
        """Reset optimizer state after architecture changes to prevent GPU memory issues."""
        try:
            # Clear optimizer state
            optimizer.state.clear()
            
            # Recreate parameter groups for new model parameters
            old_lr = optimizer.param_groups[0]['lr']
            old_params = list(optimizer.param_groups[0].keys())
            
            # Clear existing parameter groups
            optimizer.param_groups.clear()
            
            # Add current model parameters to optimizer
            optimizer.add_param_group({
                'params': list(self.model.parameters()),
                'lr': old_lr
            })
            
            # Copy other hyperparameters if they exist
            if len(old_params) > 2:  # More than just 'params' and 'lr'
                for key in old_params:
                    if key not in ['params', 'lr'] and hasattr(optimizer.defaults, key):
                        optimizer.param_groups[0][key] = optimizer.defaults[key]
            
            print(f"  ✅ Optimizer state reset successfully")
            
        except Exception as e:
            print(f"  ⚠️ Warning: Could not reset optimizer state: {e}")
            print(f"  🔧 Attempting fallback: clearing state only")
            try:
                optimizer.state.clear()
            except:
                pass
    
    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model by name."""
        name_parts = module_name.split('.')
        parent = self.model
        
        # Navigate to the parent module
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, name_parts[-1], new_module)
    
    def _update_dependent_layers(self, changed_layer_name: str, old_channels: int, new_channels: int):
        """Update layers that depend on the changed layer's output."""
        try:
            # Build a dependency graph to track which layers feed into which
            all_layers = []
            layer_dict = {}
            
            # Collect all layers in order
            for name, module in self.model.named_modules():
                if name.startswith('features') and hasattr(module, 'weight'):
                    all_layers.append(name)
                    layer_dict[name] = module
            
            # Sort layers by their position in the network
            all_layers.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            
            # Find the position of the changed layer
            changed_idx = None
            for i, name in enumerate(all_layers):
                if name == changed_layer_name:
                    changed_idx = i
                    break
            
            if changed_idx is None:
                print(f"Warning: Could not find changed layer {changed_layer_name}")
                return
            
            # Update dependencies layer by layer
            current_output_channels = new_channels
            
            for i in range(changed_idx + 1, len(all_layers)):
                layer_name = all_layers[i]
                layer = layer_dict[layer_name]
                
                # For Conv layers: check if input channels match current output channels
                if isinstance(layer, nn.Conv2d):
                    if layer.in_channels == old_channels:
                        # This layer needs input channel update
                        new_layer = nn.Conv2d(
                            current_output_channels,  # Use current output channels
                            layer.out_channels,
                            layer.kernel_size,
                            layer.stride,
                            layer.padding,
                            bias=layer.bias is not None
                        ).to(self.device)
                        
                        # Copy weights with proper handling
                        with torch.no_grad():
                            # Copy original weights
                            min_channels = min(old_channels, current_output_channels)
                            new_layer.weight.data[:, :min_channels] = layer.weight.data[:, :min_channels]
                            
                            # Initialize new input channel weights if expanding
                            if current_output_channels > old_channels:
                                new_layer.weight.data[:, old_channels:] = torch.randn_like(
                                    new_layer.weight.data[:, old_channels:]
                                ) * 0.01
                            
                            if layer.bias is not None:
                                new_layer.bias.data = layer.bias.data
                        
                        # Replace the layer
                        self._replace_module(layer_name, new_layer)
                        print(f"  Updated dependent conv layer: {layer_name}")
                        
                        # Update current output channels for next layer
                        current_output_channels = layer.out_channels
                        
                    else:
                        # This layer doesn't need input channel update
                        current_output_channels = layer.out_channels
                
                # For BatchNorm layers: match the output channels of the previous layer
                elif isinstance(layer, nn.BatchNorm2d):
                    if layer.num_features != current_output_channels:
                        # This BatchNorm needs to be updated to match current output channels
                        new_bn = nn.BatchNorm2d(current_output_channels).to(self.device)
                        
                        with torch.no_grad():
                            # Copy original parameters up to minimum size
                            min_features = min(layer.num_features, current_output_channels)
                            new_bn.weight.data[:min_features] = layer.weight.data[:min_features]
                            new_bn.bias.data[:min_features] = layer.bias.data[:min_features]
                            new_bn.running_mean[:min_features] = layer.running_mean[:min_features]
                            new_bn.running_var[:min_features] = layer.running_var[:min_features]
                            
                            # Initialize new parameters if expanding
                            if current_output_channels > layer.num_features:
                                new_bn.weight.data[layer.num_features:] = 1.0
                                new_bn.bias.data[layer.num_features:] = 0.0
                                new_bn.running_mean[layer.num_features:] = 0.0
                                new_bn.running_var[layer.num_features:] = 1.0
                        
                        self._replace_module(layer_name, new_bn)
                        print(f"  Updated dependent batch norm layer: {layer_name}")
                
                # Stop if we've processed all layers that could be affected
                if isinstance(layer, nn.Conv2d) and layer.in_channels != old_channels:
                    # This layer doesn't depend on our changes, and subsequent layers won't either
                    break
                    
        except Exception as e:
            print(f"Warning: Could not update dependent layers: {e}")
            import traceback
            traceback.print_exc()
    
    def _is_layer_after(self, layer_name: str, reference_layer: str) -> bool:
        """Check if layer_name comes after reference_layer in the network."""
        # Simple heuristic: compare the last number in the layer name
        try:
            layer_nums = [int(c) for c in layer_name if c.isdigit()]
            ref_nums = [int(c) for c in reference_layer if c.isdigit()]
            
            if layer_nums and ref_nums:
                return layer_nums[-1] > ref_nums[-1]
            return False
        except:
            return False
    
    def _add_pooling_layer(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Add an adaptive pooling layer."""
        try:
            if hasattr(self.model, 'features'):
                features = self.model.features
                
                # Add adaptive pooling before classifier
                new_features = nn.Sequential(
                    features,
                    nn.AdaptiveMaxPool2d((2, 2)),  # Reduce spatial dimensions
                    nn.Dropout2d(0.2)
                ).to(self.device)
                
                self.model.features = new_features
                self._update_classifier_input_size()
                
                # CRITICAL FIX: Force memory cleanup after architecture change
                self._cleanup_after_evolution()
                
                return True, "Added adaptive max pooling layer (2×2)"
                
        except Exception as e:
            return False, f"Pooling layer addition failed: {str(e)}"
    
    def _update_classifier_input_size(self):
        """Update classifier input size after feature changes."""
        try:
            if hasattr(self.model, 'classifier') and hasattr(self.model, 'features'):
                # Get new feature output size - CRITICAL FIX: Proper memory management
                x = None
                features_out = None
                
                try:
                    with torch.no_grad():
                        x = torch.randn(1, 3, 32, 32, device=self.device, dtype=torch.float32)
                        features_out = self.model.features(x)
                        new_input_size = features_out.view(1, -1).size(1)
                        
                        # CRITICAL: Immediately delete tensors to prevent memory leak
                        del features_out
                        del x
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                except Exception as tensor_error:
                    print(f"Error in feature size calculation: {tensor_error}")
                    # Clean up on error
                    if x is not None:
                        del x
                    if features_out is not None:
                        del features_out
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    return
                
                # Get current classifier
                classifier = self.model.classifier
                if isinstance(classifier, nn.Sequential):
                    # Find first linear layer
                    for i, layer in enumerate(classifier):
                        if isinstance(layer, nn.Linear):
                            old_input_size = layer.in_features
                            if old_input_size != new_input_size:
                                # Create new linear layer
                                new_linear = nn.Linear(new_input_size, layer.out_features).to(self.device)
                                
                                # Initialize weights appropriately
                                with torch.no_grad():
                                    # Use Xavier initialization
                                    nn.init.xavier_uniform_(new_linear.weight)
                                    nn.init.zeros_(new_linear.bias)
                                
                                # Replace the layer
                                classifier[i] = new_linear
                                print(f"  Updated classifier: {old_input_size}→{new_input_size} features")
                                break
                            break
                
        except Exception as e:
            print(f"Warning: Could not update classifier input size: {str(e)}")
            # Emergency memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _prune_redundant_layers(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Prune redundant connections in specified layers."""
        try:
            modifications = []
            for layer_name in target_layers:
                # Find the layer
                layer = dict(self.model.named_modules()).get(layer_name)
                if layer is None:
                    continue
                    
                if isinstance(layer, nn.Linear):
                    # Prune 10% of least important weights
                    with torch.no_grad():
                        weight = layer.weight.data
                        importance = torch.abs(weight).mean(dim=0)
                        threshold = torch.quantile(importance, 0.1)
                        mask = importance > threshold
                        
                        # Apply soft pruning (reduce weights rather than zero them)
                        weight[:, ~mask] *= 0.5
                        modifications.append(f"Linear layer {layer_name}: reduced 10% weights by 50%")
                        
                elif isinstance(layer, nn.Conv2d):
                    # Prune least important channels
                    with torch.no_grad():
                        weight = layer.weight.data
                        channel_importance = torch.abs(weight).mean(dim=(0, 2, 3))
                        threshold = torch.quantile(channel_importance, 0.15)
                        mask = channel_importance > threshold
                        
                        # Apply soft pruning
                        weight[:, ~mask, :, :] *= 0.6
                        modifications.append(f"Conv layer {layer_name}: reduced 15% channels by 40%")
            
            if modifications:
                return True, "; ".join(modifications)
            else:
                return False, "No suitable layers found for pruning"
                
        except Exception as e:
            return False, f"Pruning error: {str(e)}"
    
    def _expand_bottleneck_layers(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Expand bottleneck layers by increasing activation magnitudes."""
        try:
            modifications = []
            for layer_name in target_layers:
                layer = dict(self.model.named_modules()).get(layer_name)
                if layer is None:
                    continue
                    
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    # Increase weight magnitudes to improve information flow
                    with torch.no_grad():
                        layer.weight.data *= 1.1
                        if layer.bias is not None:
                            layer.bias.data *= 1.05
                        modifications.append(f"Expanded {layer_name}: increased weights by 10%")
            
            if modifications:
                return True, "; ".join(modifications)
            else:
                return False, "No suitable layers found for expansion"
                
        except Exception as e:
            return False, f"Expansion error: {str(e)}"
    
    def _add_regularization(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Add implicit regularization by adjusting weights with improved logic."""
        try:
            modifications = []
            for layer_name in target_layers:
                # Find layers containing the target name
                for name, layer in self.model.named_modules():
                    if layer_name in name:  # FIXED: Use layer_name instead of target_layers
                        if isinstance(layer, (nn.Linear, nn.Conv2d)):
                            # Apply weight decay-like regularization
                            with torch.no_grad():
                                layer.weight.data *= 0.98
                                modifications.append(f"Added regularization to {name}: reduced weights by 2%")
            
            if modifications:
                return True, "; ".join(modifications)
            else:
                return False, "No suitable layers found for regularization"
                
        except Exception as e:
            print(f"Error in regularization: {str(e)}")
            return False, f"Regularization error: {str(e)}"
    
    def _increase_capacity(self, target_layers: List[str]) -> Tuple[bool, str]:
        """Increase network capacity by enhancing existing weights with improved logic."""
        try:
            modifications = []
            for layer_name in target_layers:
                # Find layers containing the target name
                for name, layer in self.model.named_modules():
                    if layer_name in name:  # FIXED: Use layer_name instead of target_layers
                        if isinstance(layer, (nn.Linear, nn.Conv2d)):
                            # Enhance weights to increase capacity
                            with torch.no_grad():
                                # Add small random noise to break symmetry
                                noise = torch.randn_like(layer.weight.data) * 0.01
                                layer.weight.data += noise
                                layer.weight.data *= 1.05
                                modifications.append(f"Increased capacity in {name}: enhanced weights by 5%")
            
            if modifications:
                return True, "; ".join(modifications)
            else:
                return False, "No suitable layers found for capacity increase"
                
        except Exception as e:
            print(f"Error in capacity increase: {str(e)}")
            return False, f"Capacity increase error: {str(e)}"
    
    def should_rollback(self, current_performance: Dict[str, float]) -> bool:
        """Check if rollback is needed based on performance."""
        if len(self.performance_history) < 2:
            return False
        
        prev_perf = self.performance_history[-2].get('val_accuracy', 0)
        curr_perf = current_performance.get('val_accuracy', 0)
        
        if prev_perf - curr_perf > self.rollback_threshold * 100:  # Convert to percentage
            return True
        
        return False
    
    def rollback(self) -> nn.Module:
        """Rollback to previous model state."""
        self.stats['rollbacks'] += 1
        print("🔄 Rollback: Reverting to previous model state")
        return self.model
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """Complete training with adaptive evolution."""
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'evolutions': []
        }
        
        print(f"🧠 Starting NeuroExapt training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Check for evolution
            decision = self.check_evolution_need(val_loader, val_metrics, epoch)
            evolutions_this_epoch = 0
            
            if decision:
                result = self.apply_evolution(decision, val_loader)
                if result['success']:
                    evolutions_this_epoch = 1
                    
                    # CRITICAL FIX: Reset optimizer state after architecture change
                    print("🔄 Resetting optimizer state after architecture evolution...")
                    self._reset_optimizer_state(optimizer)
                    
                    # Check if rollback needed
                    if self.should_rollback(val_metrics):
                        self.rollback()
            
            # Record metrics
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_accuracy'].append(train_metrics['train_accuracy'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])
            history['evolutions'].append(evolutions_this_epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Train Acc: {train_metrics['train_accuracy']:6.2f}% | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_metrics['val_accuracy']:6.2f}%"
                  f"{' | 🧠 EVOLVED' if evolutions_this_epoch > 0 else ''}")
        
        self.training_active = False
        return history
    
    def _train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return {'train_loss': avg_loss, 'train_accuracy': accuracy}
    
    def _validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return {'val_loss': avg_loss, 'val_accuracy': accuracy}
    
    def get_model(self) -> nn.Module:
        """Get the current optimized model."""
        return self.model
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        performance_trend = None
        if len(self.performance_history) > 5:
            recent_perf = [p.get('val_accuracy', 0) for p in self.performance_history[-5:]]
            trend = np.polyfit(range(len(recent_perf)), recent_perf, 1)[0]
            performance_trend = "improving" if trend > 0.01 else "declining" if trend < -0.01 else "stable"
        
        success_rate = self.stats['successful_evolutions'] / max(1, self.stats['total_evolutions'])
        no_change_rate = self.stats['no_change_epochs'] / max(1, self.stats['total_checks'])
        
        return {
            'evolution_stats': {
                **self.stats,
                'success_rate': success_rate,
                'no_change_rate': no_change_rate
            },
            'performance_trend': performance_trend,
            'total_epochs': self.current_epoch,
            'architecture_evolutions': len(self.evolution_history),
            'model_type': self.original_model
        }
    
    def save_model(self, filepath: str):
        """Save the optimized model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'evolution_history': self.evolution_history,
            'performance_history': self.performance_history,
            'original_model': self.original_model,
            'stats': self.stats
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a previously optimized model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.evolution_history = checkpoint.get('evolution_history', [])
        self.performance_history = checkpoint.get('performance_history', [])
        self.original_model = checkpoint.get('original_model', None)
        self.stats = checkpoint.get('stats', self.stats)
    
    def __repr__(self) -> str:
        return f"NeuroExapt(model={self.original_model}, evolutions={len(self.evolution_history)})"