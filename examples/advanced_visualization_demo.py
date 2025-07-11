"""
Advanced Architecture Visualization Demo for Neuro Exapt

This example demonstrates the visualization capabilities for complex neural architectures
including multi-level branching, nested sub-networks, and dynamic structural changes.

Author: Neuro Exapt Team
"""

import torch
import torch.nn as nn
from neuroexapt.utils.visualization import ascii_model_graph, ModelVisualizer


class MultiLevelBranchNet(nn.Module):
    """
    Complex architecture with multiple levels of branching:
    - First level: 3 main branches
    - Second level: Each branch splits into sub-branches
    - Fusion at multiple points
    """
    
    def __init__(self):
        super().__init__()
        
        # Primary branches (Level 1)
        self.main_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.secondary_branch = nn.Sequential(
            nn.Conv2d(3, 32, 1),  # 1x1 conv for efficiency
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.attention_branch = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32),
            nn.Sigmoid()  # Attention weights
        )
        
        # Sub-branches from main_branch (Level 2)
        self.main_branch_deep = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.main_branch_shallow = nn.Sequential(
            nn.Conv2d(128, 128, 1),  # Skip connection style
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # First fusion: combine main sub-branches
        self.main_fusion = nn.Conv2d(256 + 128, 256, 1)
        
        # Second fusion: combine all branches
        self.fusion = nn.Conv2d(256 + 64 + 32, 384, 1)
        
        # Final processing
        self.final_conv = nn.Conv2d(384, 512, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        # Level 1 branches
        main_out = self.main_branch(x)
        secondary_out = self.secondary_branch(x)
        attention_out = self.attention_branch(x)
        
        # Level 2 branches from main
        main_deep = self.main_branch_deep(main_out)
        main_shallow = self.main_branch_shallow(main_out)
        
        # First fusion
        main_fused = self.main_fusion(torch.cat([main_deep, main_shallow], dim=1))
        
        # Apply attention
        secondary_out = secondary_out * attention_out
        
        # Final fusion
        fused = torch.cat([main_fused, secondary_out, attention_out], dim=1)
        fused = self.fusion(fused)
        
        # Final layers
        out = self.final_conv(fused)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out


class DynamicMultiLevelNet(nn.Module):
    """
    Even more complex architecture with dynamic paths and conditional execution.
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Entry branches
        self.branch_visual = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.branch_texture = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.branch_edge = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Mid-level branches (after first fusion)
        self.fusion_low = nn.Conv2d(48 + 32 + 16, 96, 1)
        
        # High-level branches
        self.branch_semantic = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.branch_context = nn.Sequential(
            nn.Conv2d(96, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Final fusion and classification
        self.fusion_high = nn.Conv2d(256 + 128, 512, 1)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Low-level feature extraction
        visual = self.branch_visual(x)
        texture = self.branch_texture(x)
        edge = self.branch_edge(x)
        
        # First fusion
        low_fused = torch.cat([visual, texture, edge], dim=1)
        low_fused = self.fusion_low(low_fused)
        
        # High-level feature extraction
        semantic = self.branch_semantic(low_fused)
        context = self.branch_context(low_fused)
        
        # Final fusion
        high_fused = torch.cat([semantic, context], dim=1)
        high_fused = self.fusion_high(high_fused)
        
        # Classification
        out = self.final_pool(high_fused)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out


def demo_basic_multi_branch():
    """Demo basic multi-level branching architecture."""
    print("\n" + "="*100)
    print("DEMO 1: Multi-Level Branching Architecture")
    print("This shows a network with branches that further split into sub-branches")
    print("="*100)
    
    model = MultiLevelBranchNet()
    sample_input = torch.randn(1, 3, 32, 32)
    
    ascii_model_graph(model, sample_input=sample_input)


def demo_dynamic_architecture():
    """Demo dynamic multi-path architecture."""
    print("\n" + "="*100)
    print("DEMO 2: Dynamic Multi-Path Architecture")
    print("This shows a network with multiple fusion points and conditional paths")
    print("="*100)
    
    model = DynamicMultiLevelNet(num_classes=100)
    sample_input = torch.randn(1, 3, 64, 64)
    
    ascii_model_graph(model, sample_input=sample_input)


def demo_architecture_evolution():
    """Demo architecture evolution with structural changes."""
    print("\n" + "="*100)
    print("DEMO 3: Architecture Evolution Visualization")
    print("This shows how the visualization handles structural changes between models")
    print("="*100)
    
    # Original model
    original = DynamicMultiLevelNet(num_classes=10)
    
    # Modified model (simulate structural evolution)
    evolved = DynamicMultiLevelNet(num_classes=10)
    
    # Add extra layers to evolved model
    evolved.branch_attention = nn.Sequential(
        nn.Conv2d(96, 48, 1),
        nn.BatchNorm2d(48),
        nn.Sigmoid()
    )
    
    # Mark changed layers
    changed_layers = [
        'fusion_low',
        'branch_semantic.2',
        'branch_semantic.3',
        'branch_attention'  # New layer
    ]
    
    sample_input = torch.randn(1, 3, 64, 64)
    
    ascii_model_graph(
        evolved, 
        previous_model=original,
        changed_layers=changed_layers,
        sample_input=sample_input
    )


def demo_custom_visualization():
    """Demo custom visualization with different terminal widths."""
    print("\n" + "="*100)
    print("DEMO 4: Custom Visualization Settings")
    print("This shows how the visualization adapts to different terminal widths")
    print("="*100)
    
    model = MultiLevelBranchNet()
    sample_input = torch.randn(1, 3, 32, 32)
    
    # Narrow terminal
    print("\n--- Narrow Terminal (80 chars) ---")
    visualizer = ModelVisualizer(terminal_width=80)
    visualizer.visualize_model(
        model, 
        sample_input=sample_input,
        title="Compact View"
    )
    
    # Wide terminal
    print("\n--- Wide Terminal (140 chars) ---")
    visualizer = ModelVisualizer(terminal_width=140)
    visualizer.visualize_model(
        model,
        sample_input=sample_input,
        title="Expanded View"
    )


def demo_extreme_complexity():
    """Demo visualization of extremely complex architecture."""
    print("\n" + "="*100)
    print("DEMO 5: Extreme Complexity Handling")
    print("This shows how the visualization handles very complex architectures gracefully")
    print("="*100)
    
    class ExtremeNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Create 5 primary branches
            for i in range(5):
                setattr(self, f'branch_{i}', nn.Sequential(
                    nn.Conv2d(3, 16 * (i + 1), 3, padding=1),
                    nn.BatchNorm2d(16 * (i + 1)),
                    nn.ReLU()
                ))
            
            # Multiple fusion layers
            self.fusion_1 = nn.Conv2d(16 + 32, 64, 1)
            self.fusion_2 = nn.Conv2d(48 + 64, 128, 1)
            self.fusion_3 = nn.Conv2d(80 + 64, 256, 1)
            self.fusion_final = nn.Conv2d(128 + 256, 512, 1)
            
            self.classifier = nn.Linear(512, 1000)
    
    model = ExtremeNet()
    ascii_model_graph(model)


def main():
    """Run all visualization demos."""
    print("\n" + "🎨 " * 20)
    print("ADVANCED NEURO EXAPT VISUALIZATION DEMONSTRATIONS")
    print("Showcasing Complex Neural Architecture Visualization")
    print("🎨 " * 20)
    
    # Run demos
    demo_basic_multi_branch()
    
    print("\n\nPress Enter to continue to next demo...")
    input()
    
    demo_dynamic_architecture()
    
    print("\n\nPress Enter to continue to next demo...")
    input()
    
    demo_architecture_evolution()
    
    print("\n\nPress Enter to continue to next demo...")
    input()
    
    demo_custom_visualization()
    
    print("\n\nPress Enter to continue to final demo...")
    input()
    
    demo_extreme_complexity()
    
    print("\n" + "="*100)
    print("✅ All demos completed!")
    print("\nKey Features Demonstrated:")
    print("1. Multi-level branching with perfect alignment")
    print("2. Multiple fusion points with clear data flow")
    print("3. Architecture evolution tracking")
    print("4. Adaptive layout for different terminal sizes")
    print("5. Graceful handling of extreme complexity")
    print("\nThe visualization module provides pixel-perfect alignment")
    print("for arbitrarily complex neural architectures!")
    print("="*100)


if __name__ == "__main__":
    main() 