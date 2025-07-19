"""
@defgroup group_visualization Visualization
@ingroup core
Visualization module for NeuroExapt framework.

Advanced visualization utilities for Neuro Exapt with automatic layout system.

This module provides sophisticated visualization capabilities for neural network
architectures, with particular emphasis on multi-branch architectures and
dynamic structural changes.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import defaultdict
import math

if TYPE_CHECKING:
    # from neuroexapt.core import ArchitectureTracker
    ArchitectureTracker = Any  # Type hint placeholder


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal visualization."""
    GREEN = '\033[92m'      # New layers
    RED = '\033[91m'        # Removed layers
    YELLOW = '\033[93m'     # Modified layers
    BLUE = '\033[94m'       # Normal layers
    CYAN = '\033[96m'       # Dimension info
    MAGENTA = '\033[95m'    # Arrows
    GRAY = '\033[90m'       # Parameter info
    WHITE = '\033[97m'      # Default text
    RESET = '\033[0m'       # Reset
    BOLD = '\033[1m'        # Bold
    STRIKETHROUGH = '\033[9m'  # Strikethrough
    DIM = '\033[2m'         # Dim text
    

@dataclass
class LayoutNode:
    """Represents a node in the layout system."""
    name: str
    width: int
    height: int
    x: int = 0
    y: int = 0
    content: List[str] = field(default_factory=list)
    children: List['LayoutNode'] = field(default_factory=list)
    parent: Optional['LayoutNode'] = None
    alignment: str = 'center'  # 'left', 'center', 'right'
    
    def add_child(self, child: 'LayoutNode'):
        """Add a child node and set parent relationship."""
        child.parent = self
        self.children.append(child)
        
    def get_absolute_position(self) -> Tuple[int, int]:
        """Get absolute position by traversing up the parent chain."""
        x, y = self.x, self.y
        node = self.parent
        while node:
            x += node.x
            y += node.y
            node = node.parent
        return x, y
    
    def get_center_x(self) -> int:
        """Get the x-coordinate of the node's center."""
        abs_x, _ = self.get_absolute_position()
        return abs_x + self.width // 2


class AutoLayout:
    """Automatic layout system for architecture visualization."""
    
    def __init__(self, terminal_width: int = 120):
        self.terminal_width = terminal_width
        self.min_spacing = 4
        self.branch_spacing = 8
        self.vertical_spacing = 1
        self._ansi_pattern = None  # Cache regex pattern
        
    def calculate_branch_layout(self, branches: Dict[str, List[Any]], 
                              additional_width: int = 0) -> Dict[str, LayoutNode]:
        """
        Calculate optimal layout for branch architecture.
        
        Args:
            branches: Dictionary mapping branch names to layer lists
            additional_width: Additional width to consider for each branch
            
        Returns:
            Dictionary mapping branch names to LayoutNode objects
        """
        branch_nodes = {}
        num_branches = len(branches)
        
        if num_branches == 0:
            return branch_nodes
        
        # Calculate width needed for each branch
        branch_widths = {}
        for branch_name, layers in branches.items():
            # Find the maximum width needed for this branch
            max_width = len(branch_name) + 4  # Branch name + padding
            
            for layer in layers:
                # Estimate layer display width
                layer_width = self._estimate_layer_width(layer) + additional_width
                max_width = max(max_width, layer_width)
            
            branch_widths[branch_name] = max_width
        
        # Calculate total width needed
        total_width = sum(branch_widths.values()) + (num_branches - 1) * self.branch_spacing
        
        # If total width exceeds terminal width, scale down
        if total_width > self.terminal_width - 10:  # Leave some margin
            scale_factor = (self.terminal_width - 10) / total_width
            branch_widths = {k: max(10, int(v * scale_factor)) for k, v in branch_widths.items()}
            total_width = sum(branch_widths.values()) + (num_branches - 1) * self.branch_spacing
        
        # Calculate starting position to center the layout
        start_x = (self.terminal_width - total_width) // 2
        
        # Create layout nodes for each branch
        current_x = start_x
        for branch_name in sorted(branches.keys()):  # Sort for consistent ordering
            width = branch_widths[branch_name]
            
            # Create the branch node
            node = LayoutNode(
                name=branch_name,
                width=width,
                height=1,  # Will be calculated based on content
                x=current_x,
                y=0
            )
            
            branch_nodes[branch_name] = node
            current_x += width + self.branch_spacing
        
        return branch_nodes
    
    def _estimate_layer_width(self, layer_info: Any) -> int:
        """Estimate the display width needed for a layer."""
        # This is a simplified estimation
        # In practice, you'd calculate based on actual layer info
        base_width = 20  # Minimum width for layer display
        
        if hasattr(layer_info, '__len__'):
            base_width = max(base_width, len(str(layer_info)))
        
        return base_width
    
    def create_trident_connection(self, source_pos: Tuple[int, int], 
                                branch_positions: List[Tuple[int, int]],
                                style: str = 'smooth') -> List[str]:
        """
        Create a trident (multi-branch) connection from source to multiple targets.
        
        Args:
            source_pos: (x, y) position of the source
            branch_positions: List of (x, y) positions for each branch
            style: Connection style ('smooth', 'angular')
            
        Returns:
            List of strings representing the connection lines
        """
        lines = []
        source_x, source_y = source_pos
        
        # Sort branches by x position for cleaner routing
        sorted_branches = sorted(branch_positions, key=lambda p: p[0])
        
        # Calculate the junction point (where the trident splits)
        # For odd number of branches, use the middle branch position
        # For even number, use the average of middle positions
        if len(sorted_branches) == 3:
            # For three branches, align with the middle branch
            junction_x = sorted_branches[1][0]
        else:
            # For other cases, calculate the center
            min_x = min(pos[0] for pos in sorted_branches)
            max_x = max(pos[0] for pos in sorted_branches)
            junction_x = (min_x + max_x) // 2
        
        junction_y = source_y + 2
        
        # Create the connection lines
        # First, vertical line from source to junction - align with junction_x
        lines.append(self._create_line_string(
            ' ' * junction_x + f"{Colors.MAGENTA}│{Colors.RESET}"
        ))
        
        # Create the split
        if len(sorted_branches) == 2:
            # Two-way split
            left_x, _ = sorted_branches[0]
            right_x, _ = sorted_branches[1]
            
            # Junction line
            junction_line = [' '] * self.terminal_width
            junction_line[left_x] = '┌'
            for i in range(left_x + 1, right_x):
                junction_line[i] = '─'
            junction_line[right_x] = '┐'
            # Place the junction symbol at the correct position
            junction_line[junction_x] = '┴'
            
            lines.append(self._create_line_string(
                f"{Colors.MAGENTA}{''.join(junction_line[:self.terminal_width])}{Colors.RESET}"
            ))
            
        elif len(sorted_branches) == 3:
            # Three-way split (true trident)
            left_x, _ = sorted_branches[0]
            mid_x, _ = sorted_branches[1]
            right_x, _ = sorted_branches[2]
            
            # Junction line
            junction_line = [' '] * self.terminal_width
            junction_line[left_x] = '┌'
            for i in range(left_x + 1, mid_x):
                junction_line[i] = '─'
            junction_line[mid_x] = '┼'
            for i in range(mid_x + 1, right_x):
                junction_line[i] = '─'
            junction_line[right_x] = '┐'
            
            lines.append(self._create_line_string(
                f"{Colors.MAGENTA}{''.join(junction_line[:self.terminal_width])}{Colors.RESET}"
            ))
            
        else:
            # Multi-way split
            junction_line = [' '] * self.terminal_width
            
            for i, (x, _) in enumerate(sorted_branches):
                if i == 0:
                    junction_line[x] = '┌'
                elif i == len(sorted_branches) - 1:
                    junction_line[x] = '┐'
                else:
                    junction_line[x] = '┬'
            
            # Fill in the horizontal lines
            for i in range(len(sorted_branches) - 1):
                start_x = sorted_branches[i][0]
                end_x = sorted_branches[i + 1][0]
                for j in range(start_x + 1, end_x):
                    if junction_line[j] == ' ':
                        junction_line[j] = '─'
            
            lines.append(self._create_line_string(
                f"{Colors.MAGENTA}{''.join(junction_line[:self.terminal_width])}{Colors.RESET}"
            ))
        
        # Vertical lines to each branch
        vert_line = [' '] * self.terminal_width
        for x, _ in sorted_branches:
            vert_line[x] = '│'
        
        lines.append(self._create_line_string(
            f"{Colors.MAGENTA}{''.join(vert_line[:self.terminal_width])}{Colors.RESET}"
        ))
        
        # Arrows pointing down
        arrow_line = [' '] * self.terminal_width
        for x, _ in sorted_branches:
            arrow_line[x] = '↓'
        
        lines.append(self._create_line_string(
            f"{Colors.MAGENTA}{''.join(arrow_line[:self.terminal_width])}{Colors.RESET}"
        ))
        
        return lines
    
    def create_fusion_connection(self, branch_positions: List[Tuple[int, int]], 
                               target_pos: Tuple[int, int]) -> List[str]:
        """
        Create a fusion connection from multiple branches to a single target.
        
        Args:
            branch_positions: List of (x, y) positions for each branch
            target_pos: (x, y) position of the target
            
        Returns:
            List of strings representing the connection lines
        """
        lines = []
        target_x, target_y = target_pos
        
        # Sort branches by x position
        sorted_branches = sorted(branch_positions, key=lambda p: p[0])
        
        # Arrows from each branch
        arrow_line = [' '] * self.terminal_width
        for x, _ in sorted_branches:
            arrow_line[x] = '↓'
        
        lines.append(self._create_line_string(
            f"{Colors.MAGENTA}{''.join(arrow_line[:self.terminal_width])}{Colors.RESET}"
        ))
        
        # Create convergence lines
        if len(sorted_branches) == 2:
            # Two branches merging
            left_x, _ = sorted_branches[0]
            right_x, _ = sorted_branches[1]
            
            merge_line = [' '] * self.terminal_width
            merge_line[left_x] = '└'
            for i in range(left_x + 1, target_x):
                merge_line[i] = '─'
            merge_line[target_x] = '┬'
            for i in range(target_x + 1, right_x):
                merge_line[i] = '─'
            merge_line[right_x] = '┘'
            
            lines.append(self._create_line_string(
                f"{Colors.MAGENTA}{''.join(merge_line[:self.terminal_width])}{Colors.RESET}"
            ))
            
        elif len(sorted_branches) == 3:
            # Three branches merging
            left_x, _ = sorted_branches[0]
            mid_x, _ = sorted_branches[1]
            right_x, _ = sorted_branches[2]
            
            merge_line = [' '] * self.terminal_width
            merge_line[left_x] = '└'
            for i in range(left_x + 1, mid_x):
                merge_line[i] = '─'
            merge_line[mid_x] = '┴'
            for i in range(mid_x + 1, right_x):
                merge_line[i] = '─'
            merge_line[right_x] = '┘'
            
            # Adjust if target is at mid position
            if target_x == mid_x:
                merge_line[mid_x] = '┼'
            
            lines.append(self._create_line_string(
                f"{Colors.MAGENTA}{''.join(merge_line[:self.terminal_width])}{Colors.RESET}"
            ))
            
        else:
            # Multiple branches merging
            merge_line = [' '] * self.terminal_width
            
            for i, (x, _) in enumerate(sorted_branches):
                if i == 0:
                    merge_line[x] = '└'
                elif i == len(sorted_branches) - 1:
                    merge_line[x] = '┘'
                else:
                    merge_line[x] = '┴'
            
            # Fill in horizontal lines
            min_x = sorted_branches[0][0]
            max_x = sorted_branches[-1][0]
            for i in range(min_x + 1, max_x):
                if merge_line[i] == ' ':
                    merge_line[i] = '─'
            
            # Mark the fusion point
            if min_x <= target_x <= max_x:
                merge_line[target_x] = '┼'
            
            lines.append(self._create_line_string(
                f"{Colors.MAGENTA}{''.join(merge_line[:self.terminal_width])}{Colors.RESET}"
            ))
        
        # Vertical line down from fusion point
        vert_line = [' '] * self.terminal_width
        vert_line[target_x] = '│'
        lines.append(self._create_line_string(
            f"{Colors.MAGENTA}{''.join(vert_line[:self.terminal_width])}{Colors.RESET}"
        ))
        
        return lines
    
    def _create_line_string(self, content: str) -> str:
        """Create a properly formatted line string."""
        if len(content) > self.terminal_width:
            return content[:self.terminal_width]
        return content
    
    def center_text(self, text: str, width: int, fill_char: str = ' ') -> str:
        """Center text within a given width."""
        text_len = len(self._strip_ansi(text))
        if text_len >= width:
            return text
        
        left_pad = (width - text_len) // 2
        right_pad = width - text_len - left_pad
        
        return fill_char * left_pad + text + fill_char * right_pad
    
    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI color codes from text for length calculation."""
        import re
        if self._ansi_pattern is None:
            self._ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return self._ansi_pattern.sub('', text)


class ModelVisualizer:
    """Enhanced model visualizer with automatic layout support."""
    
    def __init__(self, terminal_width: int = 120):
        self.layout = AutoLayout(terminal_width)
        self.terminal_width = terminal_width
        self.architecture_tracker = None
        
    def visualize_model(self, model: nn.Module, 
                       previous_model: Optional[nn.Module] = None,
                       changed_layers: Optional[List[str]] = None,
                       sample_input: Optional[torch.Tensor] = None,
                       title: str = "Dynamic Architecture Visualization",
                       architecture_tracker: Optional['ArchitectureTracker'] = None,
                       force_show: bool = False) -> None:
        """
        Visualize model architecture with automatic layout.
        
        Args:
            model: Current model to visualize
            previous_model: Previous model for comparison
            changed_layers: List of changed layer names
            sample_input: Sample input for shape inference
            title: Title for the visualization
            architecture_tracker: Optional architecture tracker for accurate change detection
        """
        if changed_layers is None:
            changed_layers = []
        
        # Store architecture tracker for use in layer info methods
        self.architecture_tracker = architecture_tracker
        
        # Check if visualization should be shown
        if not force_show and not self._should_show_visualization(model, previous_model, changed_layers):
            return
        
        # Print header
        self._print_header(title)
        
        # Analyze model structure
        structure_info = self._analyze_model_structure(model, previous_model, changed_layers)
        
        if structure_info['is_multi_branch']:
            self._visualize_multi_branch(model, structure_info, previous_model, 
                                       changed_layers, sample_input)
        else:
            self._visualize_sequential(model, structure_info, previous_model,
                                      changed_layers, sample_input)
        
        # Print summary
        self._print_summary(structure_info)
        
        # Reset layer statuses after visualization
        if architecture_tracker:
            architecture_tracker.reset_status()
    
    def _should_show_visualization(self, model: nn.Module, previous_model: Optional[nn.Module], changed_layers: List[str]) -> bool:
        """Determine if visualization should be shown based on real changes."""
        if not previous_model:
            return True  # First time showing
        
        # Check for real parameter changes
        current_params = sum(p.numel() for p in model.parameters())
        previous_params = sum(p.numel() for p in previous_model.parameters())
        
        return current_params != previous_params or len(changed_layers) > 0
    
    def _print_header(self, title: str):
        """Print visualization header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}🏗️  {title}{Colors.RESET}")
        print("=" * self.terminal_width)
    
    def _analyze_model_structure(self, model: nn.Module, 
                                previous_model: Optional[nn.Module] = None,
                                changed_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze model structure and gather information."""
        current_layers = dict(model.named_modules())
        
        # Calculate parameters for each layer
        layer_params = {}
        for name, module in current_layers.items():
            if list(module.parameters()) and len(list(module.children())) == 0:
                layer_params[name] = sum(p.numel() for p in module.parameters())
        
        # Analyze previous model if provided
        prev_layers = {}
        prev_params = {}
        if previous_model:
            prev_layers = dict(previous_model.named_modules())
            for name, module in prev_layers.items():
                if list(module.parameters()) and len(list(module.children())) == 0:
                    prev_params[name] = sum(p.numel() for p in module.parameters())
        
        # Detect architecture type
        all_layer_names = list(current_layers.keys())
        is_multi_branch = any(
            name.startswith(('branch_', 'main_branch', 'secondary_branch', 'attention_branch'))
            for name in all_layer_names
        )
        
        # Group layers by branches if multi-branch
        branches = {}
        other_layers = []
        
        if is_multi_branch:
            for name in all_layer_names:
                if any(name.startswith(prefix) for prefix in 
                      ['branch_', 'main_branch', 'secondary_branch', 'attention_branch']):
                    # Extract branch name
                    branch_name = self._extract_branch_name(name)
                    if branch_name not in branches:
                        branches[branch_name] = []
                    if name in layer_params or (previous_model and name in prev_params):
                        branches[branch_name].append(name)
                elif name in layer_params or (previous_model and name in prev_params):
                    other_layers.append(name)
        else:
            other_layers = [name for name in all_layer_names 
                           if name in layer_params or (previous_model and name in prev_params)]
        
        return {
            'current_layers': current_layers,
            'prev_layers': prev_layers,
            'layer_params': layer_params,
            'prev_params': prev_params,
            'is_multi_branch': is_multi_branch,
            'branches': branches,
            'other_layers': other_layers,
            'total_params': sum(layer_params.values()),
            'prev_total_params': sum(prev_params.values()) if prev_params else 0
        }
    
    def _extract_branch_name(self, layer_name: str) -> str:
        """Extract branch name from layer name."""
        if layer_name.startswith('main_branch'):
            return 'main_branch'
        elif layer_name.startswith('secondary_branch'):
            return 'secondary_branch'
        elif layer_name.startswith('attention_branch'):
            return 'attention_branch'
        elif layer_name.startswith('branch_'):
            parts = layer_name.split('.')
            return parts[0]
        else:
            return 'unknown_branch'
    
    def _visualize_multi_branch(self, model: nn.Module, 
                               structure_info: Dict[str, Any],
                               previous_model: Optional[nn.Module],
                               changed_layers: List[str],
                               sample_input: Optional[torch.Tensor]) -> None:
        """Visualize multi-branch architecture with automatic layout."""
        print(f"{Colors.BLUE}📊 Multi-Branch Architecture{Colors.RESET}")
        print("-" * self.terminal_width)
        
        # Calculate branch layout first to determine input position
        branches = structure_info['branches']
        branch_layout = self.layout.calculate_branch_layout(branches, additional_width=10)
        
        if not branch_layout:
            return
        
        # Get branch center positions
        branch_centers = [(node.get_center_x(), 0) for node in branch_layout.values()]
        sorted_centers = sorted(branch_centers, key=lambda p: p[0])
        
        # Calculate input position - align with trident junction
        if len(sorted_centers) == 3:
            # For three branches, align with middle branch
            input_x = sorted_centers[1][0]
        else:
            # For other cases, use center of all branches
            min_x = min(pos[0] for pos in sorted_centers)
            max_x = max(pos[0] for pos in sorted_centers)
            input_x = (min_x + max_x) // 2
        
        # Show input aligned with junction
        input_shape = self._get_input_shape(sample_input)
        input_text = f"{Colors.CYAN}Input: {input_shape}{Colors.RESET}"
        print("\n" + " " * (input_x - len(self.layout._strip_ansi(input_text)) // 2) + input_text)
        
        # Create trident connection from input
        trident_lines = self.layout.create_trident_connection(
            (input_x, 0), branch_centers
        )
        
        for line in trident_lines:
            print(line)
        
        # Print branch headers with perfect alignment
        header_line = ""
        for branch_name, node in sorted(branch_layout.items()):
            branch_title = branch_name.replace('_', ' ').title()
            centered_title = self.layout.center_text(
                f"{Colors.BOLD}{branch_title}{Colors.RESET}", 
                node.width
            )
            
            # Position the title at the node's location
            # Use _strip_ansi to calculate correct position
            current_pos = len(self.layout._strip_ansi(header_line))
            padding = node.x - current_pos
            header_line += " " * max(0, padding) + centered_title
        
        print(header_line)
        print()
        
        # Visualize each layer in branches
        max_depth = max(len(branches[b]) for b in branches if b in branch_layout)
        
        for depth in range(max_depth):
            self._visualize_branch_layer(
                depth, branches, branch_layout, model, structure_info,
                previous_model, changed_layers, sample_input
            )
        
        # Calculate fusion position to pass to downstream layers
        sorted_positions = sorted([(node.get_center_x(), 0) for node in branch_layout.values()], 
                                key=lambda p: p[0])
        if len(sorted_positions) == 3:
            fusion_x = sorted_positions[1][0]
        else:
            min_x = min(pos[0] for pos in sorted_positions)
            max_x = max(pos[0] for pos in sorted_positions)
            fusion_x = (min_x + max_x) // 2
        
        # Handle fusion layers
        self._visualize_fusion(
            structure_info, branch_layout, model, previous_model,
            changed_layers, sample_input
        )
        
        # Visualize remaining layers with same alignment
        if structure_info['other_layers']:
            self._visualize_other_layers(
                structure_info, model, previous_model,
                changed_layers, sample_input, center_x=fusion_x
            )
    
    def _visualize_branch_layer(self, depth: int, branches: Dict[str, List[str]],
                               branch_layout: Dict[str, LayoutNode],
                               model: nn.Module, structure_info: Dict[str, Any],
                               previous_model: Optional[nn.Module],
                               changed_layers: List[str],
                               sample_input: Optional[torch.Tensor]) -> None:
        """Visualize a single depth level across all branches."""
        layer_line = ""
        param_line = ""
        shape_line = ""
        
        for branch_name, node in sorted(branch_layout.items()):
            branch_layers = branches.get(branch_name, [])
            
            if depth < len(branch_layers):
                layer_name = branch_layers[depth]
                
                # Get layer info
                layer_info = self._get_layer_info(
                    layer_name, model, structure_info, 
                    previous_model, changed_layers
                )
                
                # Create layer representation
                layer_repr = self._format_layer(layer_info)
                param_repr = self._format_params(layer_info)
                shape_repr = self._format_shape(layer_name, model, sample_input, layer_info)
                
                # Center within branch width
                layer_repr = self.layout.center_text(layer_repr, node.width)
                param_repr = self.layout.center_text(param_repr, node.width)
                shape_repr = self.layout.center_text(shape_repr, node.width)
                
                # Add to lines with proper positioning
                padding = node.x - len(self.layout._strip_ansi(layer_line))
                layer_line += " " * max(0, padding) + layer_repr
                
                padding = node.x - len(self.layout._strip_ansi(param_line))
                param_line += " " * max(0, padding) + param_repr
                
                padding = node.x - len(self.layout._strip_ansi(shape_line))
                shape_line += " " * max(0, padding) + shape_repr
        
        # Only print if we have actual content
        has_content = any(branches.get(branch_name, []) and depth < len(branches[branch_name]) 
                         for branch_name in branch_layout.keys())
        
        if has_content:
            # Print layer lines only if there's content
            if layer_line.strip():
                print(layer_line.rstrip())
            if param_line.strip():
                print(param_line.rstrip())
            if shape_line.strip():
                print(shape_line.rstrip())
            
            # Add vertical arrows if not last layer
            if depth < max(len(branches.get(b, [])) for b in branches) - 1:
                arrow_line = ""
                for branch_name, node in sorted(branch_layout.items()):
                    branch_layers = branches.get(branch_name, [])
                    if branch_layers and depth < len(branch_layers) - 1:
                        arrow = f"{Colors.MAGENTA}↓{Colors.RESET}"
                        arrow_centered = self.layout.center_text(arrow, node.width)
                        padding = node.x - len(self.layout._strip_ansi(arrow_line))
                        arrow_line += " " * max(0, padding) + arrow_centered
                
                if arrow_line.strip():
                    print(arrow_line.rstrip())
            
            # Only add blank line if we printed something
            print()
    
    def _visualize_fusion(self, structure_info: Dict[str, Any],
                         branch_layout: Dict[str, LayoutNode],
                         model: nn.Module, previous_model: Optional[nn.Module],
                         changed_layers: List[str],
                         sample_input: Optional[torch.Tensor]) -> None:
        """Visualize fusion of branches."""
        # Find fusion layers
        fusion_layers = [layer for layer in structure_info['other_layers']
                        if 'fusion' in layer.lower()]
        
        if not fusion_layers or not branch_layout:
            return
        
        # Remove the redundant "Branch Fusion:" line
        print()  # Just add a blank line for spacing
        
        # Show branch outputs
        output_line = ""
        branches = structure_info['branches']
        
        for branch_name, node in sorted(branch_layout.items()):
            if branch_name in branches and branches[branch_name]:
                last_layer = branches[branch_name][-1]
                output_shape = self._get_shape_annotation(last_layer, model, sample_input)
                
                # Check if shape changed
                if previous_model and last_layer in structure_info['prev_layers']:
                    prev_shape = self._get_shape_annotation(last_layer, previous_model, sample_input)
                    if prev_shape != output_shape:
                        output_shape = f"{Colors.YELLOW}{output_shape}{Colors.RESET}"
                    else:
                        output_shape = f"{Colors.CYAN}{output_shape}{Colors.RESET}"
                else:
                    output_shape = f"{Colors.CYAN}{output_shape}{Colors.RESET}"
                
                centered_shape = self.layout.center_text(output_shape, node.width)
                padding = node.x - len(self.layout._strip_ansi(output_line))
                output_line += " " * max(0, padding) + centered_shape
        
        print(output_line.rstrip())
        
        # Create fusion connection
        branch_positions = [(node.get_center_x(), 0) for node in branch_layout.values()]
        sorted_positions = sorted(branch_positions, key=lambda p: p[0])
        
        # Calculate fusion position - same logic as input position
        if len(sorted_positions) == 3:
            # For three branches, align with middle branch
            fusion_x = sorted_positions[1][0]
        else:
            # For other cases, use center of all branches
            min_x = min(pos[0] for pos in sorted_positions)
            max_x = max(pos[0] for pos in sorted_positions)
            fusion_x = (min_x + max_x) // 2
        
        fusion_lines = self.layout.create_fusion_connection(
            branch_positions, (fusion_x, 0)
        )
        
        for line in fusion_lines:
            print(line)
        
        # Show fusion layer with fusion method information
        for fusion_layer in fusion_layers:
            layer_info = self._get_layer_info(
                fusion_layer, model, structure_info,
                previous_model, changed_layers
            )
            
            # Get the actual layer module to determine fusion method
            layer_module = self._get_layer_module(fusion_layer, model)
            fusion_method = self._get_fusion_method(layer_module, fusion_layer)
            
            # Format layer with fusion method
            layer_repr = self._format_layer(layer_info)
            if fusion_method:
                layer_repr += f" {Colors.DIM}[{fusion_method}]{Colors.RESET}"
            
            param_repr = self._format_params(layer_info)
            
            # Display fusion layer aligned with fusion position
            layer_display = f"{layer_repr}"
            param_display = f"{param_repr}"
            
            # Calculate center position for the fusion layer display
            layer_len = len(self.layout._strip_ansi(layer_display))
            param_len = len(self.layout._strip_ansi(param_display))
            
            # Center the layer name
            layer_padding = fusion_x - layer_len // 2
            print(" " * max(0, layer_padding) + layer_display)
            
            # Center the parameters
            param_padding = fusion_x - param_len // 2  
            print(" " * max(0, param_padding) + param_display)
            
            # Place arrow directly below at fusion position
            print(" " * fusion_x + f"{Colors.MAGENTA}↓{Colors.RESET}")
            print()
    
    def _get_fusion_method(self, layer_module: Optional[nn.Module], layer_name: str) -> str:
        """Determine the fusion method based on the layer implementation."""
        if not layer_module:
            return ""
        
        # Check common fusion patterns
        if hasattr(layer_module, 'fusion_method'):
            return str(layer_module.fusion_method)
        
        # Infer from layer type or name
        layer_name_lower = layer_name.lower()
        if 'concat' in layer_name_lower or 'concatenate' in layer_name_lower:
            return "concat"
        elif 'add' in layer_name_lower or 'sum' in layer_name_lower:
            return "add"
        elif 'multiply' in layer_name_lower or 'mul' in layer_name_lower:
            return "multiply"
        elif 'attention' in layer_name_lower:
            return "attention"
        elif 'weighted' in layer_name_lower:
            return "weighted"
        
        # Check if it's a Conv layer (common for concat fusion)
        if isinstance(layer_module, nn.Conv2d):
            # If the input channels suggest concatenation
            if hasattr(layer_module, 'in_channels'):
                # This is a heuristic - if in_channels is larger than expected,
                # it might be concat fusion
                return "concat"
        
        return ""
    
    def _visualize_sequential(self, model: nn.Module,
                             structure_info: Dict[str, Any],
                             previous_model: Optional[nn.Module],
                             changed_layers: List[str],
                             sample_input: Optional[torch.Tensor]) -> None:
        """Visualize sequential architecture."""
        print(f"{Colors.BLUE}📈 Sequential Architecture{Colors.RESET}")
        print("-" * self.terminal_width)
        
        # Show input
        input_shape = self._get_input_shape(sample_input)
        input_text = f"{Colors.CYAN}Input: {input_shape}{Colors.RESET}"
        center_x = self.terminal_width // 2
        input_len = len(self.layout._strip_ansi(input_text))
        print("\n" + " " * max(0, center_x - input_len // 2) + input_text)
        print()
        
        # Visualize each layer
        layers = structure_info['other_layers']
        
        for i, layer_name in enumerate(layers):
            layer_info = self._get_layer_info(
                layer_name, model, structure_info,
                previous_model, changed_layers
            )
            
            # Format layer information
            layer_repr = self._format_layer(layer_info)
            param_repr = self._format_params(layer_info)
            shape_repr = self._format_shape(layer_name, model, sample_input, layer_info)
            
            # Center each component independently for proper alignment
            center_x = self.terminal_width // 2
            
            layer_len = len(self.layout._strip_ansi(layer_repr))
            param_len = len(self.layout._strip_ansi(param_repr))
            shape_len = len(self.layout._strip_ansi(shape_repr))
            
            print(" " * max(0, center_x - layer_len // 2) + layer_repr)
            print(" " * max(0, center_x - param_len // 2) + param_repr)
            print(" " * max(0, center_x - shape_len // 2) + shape_repr)
            
            # Add arrow if not last layer
            if i < len(layers) - 1:
                print(" " * center_x + f"{Colors.MAGENTA}↓{Colors.RESET}")
            
            print()
        
        # Show final output
        if layers:
            final_layer = layers[-1]
            final_shape = self._get_shape_annotation(final_layer, model, sample_input)
            
            # Check if shape changed
            if previous_model and final_layer in structure_info['prev_layers']:
                prev_shape = self._get_shape_annotation(final_layer, previous_model, sample_input)
                if prev_shape != final_shape:
                    final_shape = f"{Colors.YELLOW}Output: {final_shape}{Colors.RESET}"
                else:
                    final_shape = f"{Colors.CYAN}Output: {final_shape}{Colors.RESET}"
            else:
                final_shape = f"{Colors.CYAN}Output: {final_shape}{Colors.RESET}"
            
            final_len = len(self.layout._strip_ansi(final_shape))
            print(" " * max(0, center_x - final_len // 2) + final_shape)
    
    def _visualize_other_layers(self, structure_info: Dict[str, Any],
                               model: nn.Module, previous_model: Optional[nn.Module],
                               changed_layers: List[str],
                               sample_input: Optional[torch.Tensor],
                               center_x: Optional[int] = None) -> None:
        """Visualize remaining layers after branches."""
        other_layers = [layer for layer in structure_info['other_layers']
                       if 'fusion' not in layer.lower()]
        
        if not other_layers:
            return
        
        print("-" * self.terminal_width)
        
        # Use provided center_x or default to terminal center
        if center_x is None:
            center_x = self.terminal_width // 2
        
        for i, layer_name in enumerate(other_layers):
            layer_info = self._get_layer_info(
                layer_name, model, structure_info,
                previous_model, changed_layers
            )
            
            layer_repr = self._format_layer(layer_info)
            param_repr = self._format_params(layer_info)
            
            # Center at the specified position
            layer_len = len(self.layout._strip_ansi(layer_repr))
            param_len = len(self.layout._strip_ansi(param_repr))
            
            layer_padding = center_x - layer_len // 2
            print(" " * max(0, layer_padding) + layer_repr)
            
            param_padding = center_x - param_len // 2
            print(" " * max(0, param_padding) + param_repr)
            
            if i < len(other_layers) - 1:
                print(" " * center_x + f"{Colors.MAGENTA}↓{Colors.RESET}")
            
            print()
    
    def _get_layer_info(self, layer_name: str, model: nn.Module,
                       structure_info: Dict[str, Any],
                       previous_model: Optional[nn.Module],
                       changed_layers: List[str]) -> Dict[str, Any]:
        """Get comprehensive information about a layer."""
        info = {
            'name': layer_name,
            'simplified_name': self._simplify_layer_name(layer_name),
            'exists_in_current': layer_name in structure_info['current_layers'],
            'exists_in_previous': layer_name in structure_info.get('prev_layers', {}),
            'params': structure_info['layer_params'].get(layer_name, 0),
            'prev_params': structure_info.get('prev_params', {}).get(layer_name, 0),
            'changed': layer_name in changed_layers,
            'status': 'normal',
            'layer_type': None
        }
        
        # Determine layer type
        if info['exists_in_current']:
            layer = self._get_layer_module(layer_name, model)
            info['layer_type'] = type(layer).__name__ if layer else 'Unknown'
        elif info['exists_in_previous'] and previous_model:
            layer = self._get_layer_module(layer_name, previous_model)
            info['layer_type'] = type(layer).__name__ if layer else 'Unknown'
        
        # Use architecture tracker if available for accurate status
        if self.architecture_tracker:
            tracker_status = self.architecture_tracker.get_layer_status(layer_name, 'initial')
            if tracker_status == 'new':
                info['status'] = 'new'
            elif tracker_status == 'removed':
                info['status'] = 'removed'
            elif tracker_status == 'modified':
                info['status'] = 'modified'
            elif tracker_status == 'unchanged':
                info['status'] = 'normal'
        else:
            # Fallback to original logic
            if previous_model:
                if not info['exists_in_current'] and info['exists_in_previous']:
                    info['status'] = 'removed'
                elif info['exists_in_current'] and not info['exists_in_previous']:
                    info['status'] = 'new'
                elif info['changed'] or info['params'] != info['prev_params']:
                    info['status'] = 'modified'
            elif info['changed']:
                info['status'] = 'modified'
        
        return info
    
    def _get_layer_module(self, layer_name: str, model: nn.Module) -> Optional[nn.Module]:
        """Get the actual layer module from the model."""
        try:
            parts = layer_name.split('.')
            module = model
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    return None
            return module
        except:
            return None
    
    def _format_layer(self, layer_info: Dict[str, Any]) -> str:
        """Format layer representation with status and color."""
        name = layer_info['simplified_name']
        layer_type = layer_info['layer_type'] or 'Unknown'
        
        # Shorten layer type names
        type_short = layer_type.replace('Conv2d', 'Conv') \
                              .replace('Linear', 'FC') \
                              .replace('BatchNorm2d', 'BN') \
                              .replace('BatchNorm1d', 'BN')
        
        # Determine color and decoration
        if layer_info['status'] == 'removed':
            return f"{Colors.RED}✗{Colors.STRIKETHROUGH}{name} {type_short}{Colors.RESET}"
        elif layer_info['status'] == 'new':
            return f"{Colors.GREEN}✓{Colors.BOLD}{name} {type_short}{Colors.RESET}"
        elif layer_info['status'] == 'modified':
            return f"{Colors.YELLOW}◇{name} {type_short}{Colors.RESET}"
        else:
            return f"{Colors.BLUE} {name} {type_short}{Colors.RESET}"
    
    def _format_params(self, layer_info: Dict[str, Any]) -> str:
        """Format parameter count with change highlighting."""
        params = layer_info['params']
        prev_params = layer_info['prev_params']
        
        param_str = self._format_param_count(params)
        
        if layer_info['exists_in_previous'] and params != prev_params:
            return f"{Colors.YELLOW}{param_str}{Colors.RESET}"
        else:
            return f"{Colors.GRAY}{param_str}{Colors.RESET}"
    
    def _format_shape(self, layer_name: str, model: nn.Module,
                     sample_input: Optional[torch.Tensor],
                     layer_info: Dict[str, Any]) -> str:
        """Format shape information for a layer."""
        shape = self._get_shape_annotation(layer_name, model, sample_input)
        
        # Highlight if shape might have changed
        if layer_info['status'] == 'modified':
            return f"{Colors.YELLOW}{shape}{Colors.RESET}"
        else:
            return f"{Colors.CYAN}{shape}{Colors.RESET}"
    
    def _get_shape_annotation(self, layer_name: str, model: nn.Module,
                             sample_input: Optional[torch.Tensor]) -> str:
        """Get shape annotation for a layer's output."""
        try:
            layer = self._get_layer_module(layer_name, model)
            if not layer:
                return "[?]"
            
            if isinstance(layer, nn.Conv2d):
                return f"[{layer.out_channels},H,W]"
            elif isinstance(layer, nn.Linear):
                return f"[{layer.out_features}]"
            elif isinstance(layer, nn.MaxPool2d):
                return "[C,H/2,W/2]"
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                output_size = layer.output_size
                if isinstance(output_size, tuple):
                    return f"[C,{output_size[0]},{output_size[1]}]"
                else:
                    return f"[C,{output_size},{output_size}]"
            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                return f"[{layer.num_features}]"
            elif isinstance(layer, nn.Flatten):
                return "[N]"
            else:
                return "[?]"
        except:
            return "[?]"
    
    def _get_input_shape(self, sample_input: Optional[torch.Tensor]) -> str:
        """Get input shape string."""
        if sample_input is not None:
            return f"[{','.join(map(str, sample_input.shape[1:]))}]"
        return "[3,32,32]"  # Default
    
    def _simplify_layer_name(self, layer_name: str) -> str:
        """Simplify layer name for display."""
        prefixes_to_remove = ['model.', 'net.', 'backbone.', 'features.', 'classifier.']
        
        simplified = layer_name
        for prefix in prefixes_to_remove:
            if simplified.startswith(prefix):
                simplified = simplified[len(prefix):]
                break
        
        return simplified
    
    def _format_param_count(self, count: int) -> str:
        """Format parameter count in human-readable form."""
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}K"
        else:
            return str(count)
    
    def _print_summary(self, structure_info: Dict[str, Any]) -> None:
        """Print summary statistics."""
        print("\n" + "=" * self.terminal_width)
        
        total_params = structure_info['total_params']
        print(f"{Colors.BOLD}📊 Total Parameters: {self._format_param_count(total_params)}{Colors.RESET}")
        
        if structure_info['prev_total_params'] > 0:
            prev_total = structure_info['prev_total_params']
            param_change = total_params - prev_total
            
            if param_change > 0:
                print(f"{Colors.GREEN}📈 Parameter Change: +{self._format_param_count(abs(param_change))}{Colors.RESET}")
            elif param_change < 0:
                print(f"{Colors.RED}📉 Parameter Change: -{self._format_param_count(abs(param_change))}{Colors.RESET}")
            else:
                print(f"{Colors.BLUE}📊 Parameter Change: No change{Colors.RESET}")
        
        # Legend
        print(f"\n{Colors.BOLD}Legend:{Colors.RESET}")
        print(f"  {Colors.GREEN}✓ New layers{Colors.RESET}   "
              f"{Colors.RED}✗ Removed layers{Colors.RESET}   "
              f"{Colors.YELLOW}◇ Modified layers{Colors.RESET}")
        print(f"  {Colors.CYAN}[Shape] annotations{Colors.RESET}   "
              f"{Colors.GRAY}Parameter counts{Colors.RESET}   "
              f"{Colors.MAGENTA}Data flow arrows{Colors.RESET}")
        
        print("=" * self.terminal_width)


# Public API functions that maintain compatibility
def ascii_model_graph(model, previous_model=None, changed_layers=None, sample_input=None, 
                     architecture_tracker=None, force_show=False):
    """
    Create ASCII visualization of model architecture.
    
    This is the main public API function that maintains backward compatibility.
    """
    visualizer = ModelVisualizer()
    visualizer.visualize_model(model, previous_model, changed_layers, sample_input,
                             architecture_tracker=architecture_tracker, force_show=force_show)


def plot_evolution_history(evolution_history: List[Any], save_path: Optional[str] = None):
    """Plot evolution history showing structural changes over time."""
    if not evolution_history:
        return
        
    # Extract data
    epochs = [step.epoch for step in evolution_history]
    params_before = [step.parameters_before for step in evolution_history]
    params_after = [step.parameters_after for step in evolution_history]
    actions = [step.action for step in evolution_history]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot parameter evolution
    ax1.plot(epochs, params_before, 'o-', label='Before', alpha=0.7)
    ax1.plot(epochs, params_after, 's-', label='After', alpha=0.7)
    
    # Add action annotations
    for i, (epoch, action) in enumerate(zip(epochs, actions)):
        if action != 'none':
            ax1.annotate(
                action,
                xy=(epoch, params_after[i]),
                xytext=(epoch, params_after[i] + 1000),
                arrowprops=dict(arrowstyle='->', alpha=0.5),
                fontsize=8,
                ha='center'
            )
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_title('Parameter Evolution During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot efficiency gains
    efficiency_gains = [step.efficiency_gain() for step in evolution_history]
    colors = ['green' if g > 0 else 'red' for g in efficiency_gains]
    
    ax2.bar(epochs, efficiency_gains, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Efficiency Gain')
    ax2.set_title('Structural Evolution Efficiency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_entropy_history(entropy_history: List[float], 
                        threshold_history: List[float],
                        save_path: Optional[str] = None):
    """Plot entropy evolution with adaptive threshold."""
    plt.figure(figsize=(10, 6))
    
    # Plot entropy
    plt.plot(entropy_history, label='Entropy', alpha=0.8, linewidth=2)
    
    # Plot threshold
    if len(threshold_history) > len(entropy_history):
        threshold_history = threshold_history[:len(entropy_history)]
    elif len(threshold_history) < len(entropy_history):
        # Extend threshold history
        last_threshold = threshold_history[-1] if threshold_history else 0.5
        threshold_history.extend([last_threshold] * (len(entropy_history) - len(threshold_history)))
        
    plt.plot(threshold_history, '--', label='Threshold', color='red', alpha=0.8)
    
    # Add shaded regions
    plt.fill_between(
        range(len(entropy_history)),
        0,
        threshold_history,
        alpha=0.2,
        color='red',
        label='Pruning Zone'
    )
    
    plt.xlabel('Training Step')
    plt.ylabel('Entropy')
    plt.title('Adaptive Entropy Control')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_layer_importance_heatmap(layer_importances: Dict[str, List[float]],
                                 save_path: Optional[str] = None):
    """Plot heatmap of layer importances over time."""
    if not layer_importances:
        return
        
    # Prepare data
    layer_names = list(layer_importances.keys())
    importance_matrix = []
    
    max_len = max(len(hist) for hist in layer_importances.values())
    
    for name in layer_names:
        history = layer_importances[name]
        # Pad if necessary
        if len(history) < max_len:
            history = history + [history[-1]] * (max_len - len(history))
        importance_matrix.append(history)
    
    importance_matrix = np.array(importance_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        importance_matrix,
        xticklabels=False,
        yticklabels=layer_names,
        cmap='YlOrRd',
        cbar_kws={'label': 'Importance'},
        vmin=0
    )
    
    plt.xlabel('Training Step')
    plt.ylabel('Layer')
    plt.title('Layer Importance Evolution')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def plot_information_metrics(metrics_history: Dict[str, List[float]],
                           save_path: Optional[str] = None):
    """Plot information-theoretic metrics over training."""
    n_metrics = len(metrics_history)
    if n_metrics == 0:
        return
        
    fig, axes = plt.subplots(
        (n_metrics + 1) // 2,
        2,
        figsize=(12, 4 * ((n_metrics + 1) // 2))
    )
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axes[idx]
        
        ax.plot(values, alpha=0.8)
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add moving average
        if len(values) > 10:
            window = min(50, len(values) // 10)
            ma = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(values)), ma, 'r--', alpha=0.8, label='MA')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def create_summary_plot(ne_instance, save_path: Optional[str] = None):
    """Create a comprehensive summary plot of Neuro Exapt training."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Evolution history (top-left, 2x1)
    ax1 = fig.add_subplot(gs[0:2, 0])
    if ne_instance.evolution_history:
        epochs = [step.epoch for step in ne_instance.evolution_history]
        params = [step.parameters_after for step in ne_instance.evolution_history]
        ax1.plot(epochs, params, 'o-')
        ax1.set_title('Parameter Evolution')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Parameters')
    else:
        ax1.text(0.5, 0.5, 'No evolution history', ha='center', va='center')
    
    # 2. Entropy history (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if ne_instance.entropy_ctrl.entropy_history:
        ax2.plot(ne_instance.entropy_ctrl.entropy_history[-100:])
        ax2.set_title('Recent Entropy')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Entropy')
    else:
        ax2.text(0.5, 0.5, 'No entropy data', ha='center', va='center')
    
    # 3. Current metrics (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics_text = f"""Current Status:
    Epoch: {ne_instance.current_epoch}
    Entropy Threshold: {ne_instance.entropy_ctrl.threshold:.3f}
    Evolution Steps: {len(ne_instance.evolution_history)}
    """
    ax3.text(0.1, 0.8, metrics_text, va='top', fontsize=12)
    
    # 4. Layer importance (middle row, full width)
    ax4 = fig.add_subplot(gs[1, :])
    layer_importances = ne_instance.info_theory._activation_cache.get('layer_importances', {})
    if layer_importances:
        names = list(layer_importances.keys())[:20]  # Limit to 20 layers
        values = [layer_importances[n] for n in names]
        ax4.barh(names, values)
        ax4.set_xlabel('Importance')
        ax4.set_title('Current Layer Importances')
    else:
        ax4.text(0.5, 0.5, 'No layer importance data', ha='center', va='center')
    
    # 5. Evolution summary (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    if ne_instance.evolution_history:
        actions = {}
        for step in ne_instance.evolution_history:
            actions[step.action] = actions.get(step.action, 0) + 1
        
        ax5.bar(list(actions.keys()), list(actions.values()))
        ax5.set_title('Evolution Action Summary')
        ax5.set_xlabel('Action Type')
        ax5.set_ylabel('Count')
    else:
        ax5.text(0.5, 0.5, 'No evolution actions', ha='center', va='center')
    
    # Add title
    fig.suptitle(
        f'Neuro Exapt Training Summary - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        fontsize=16
    )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def beautify_complexity(complexity_dict: Dict[str, Any]) -> str:
    """
    Beautify the complexity analysis output for better readability.
    
    Args:
        complexity_dict: Dictionary from calculate_network_complexity
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Extract main metrics
    total_params = complexity_dict.get('total_parameters', 0)
    trainable_params = complexity_dict.get('trainable_parameters', 0)
    depth = complexity_dict.get('depth', 0)
    width_stats = complexity_dict.get('width_stats', {})
    flops = complexity_dict.get('estimated_flops', None)
    
    # Format main stats
    lines.append(f"   Total parameters: {total_params:,}")
    lines.append(f"   Trainable parameters: {trainable_params:,}")
    lines.append(f"   Network depth: {depth} layers")
    
    if width_stats:
        lines.append(f"   Layer width: {int(width_stats.get('mean', 0))} ± {int(width_stats.get('std', 0))} (min: {width_stats.get('min', 0)}, max: {width_stats.get('max', 0)})")
    
    if flops is not None and flops > 0:
        # Convert FLOPs to human-readable format
        if flops >= 1e9:
            lines.append(f"   Estimated FLOPs: {flops/1e9:.2f}G")
        elif flops >= 1e6:
            lines.append(f"   Estimated FLOPs: {flops/1e6:.2f}M")
        else:
            lines.append(f"   Estimated FLOPs: {flops:,}")
    
    return '\n'.join(lines)


# Convenience function for backward compatibility
def print_architecture(model, previous_model=None, changed_layers=None):
    """Print simplified architecture information."""
    if changed_layers is None:
        changed_layers = []
    
    print("🏗️  Model Architecture Overview")
    print("=" * 50)
    
    current_layers = dict(model.named_modules())
    layer_params = {
        name: sum(p.numel() for p in module.parameters()) 
        for name, module in current_layers.items() 
        if list(module.parameters()) and len(list(module.children())) == 0
    }
    
    if previous_model:
        prev_layers = dict(previous_model.named_modules())
        prev_params = {
            name: sum(p.numel() for p in module.parameters()) 
            for name, module in prev_layers.items() 
            if list(module.parameters()) and len(list(module.children())) == 0
        }
        all_layers = sorted(set(list(current_layers.keys()) + list(prev_layers.keys())))
    else:
        all_layers = sorted(layer_params.keys())
    
    visualizer = ModelVisualizer()
    
    for name in all_layers:
        if name in layer_params or (previous_model and name in prev_params):
            # Determine status
            if previous_model:
                if name not in current_layers:
                    prefix = "[REMOVED] "
                    layer_type = type(prev_layers[name]).__name__
                    params = prev_params.get(name, 0)
                elif name not in prev_layers:
                    prefix = "[NEW] "
                    layer_type = type(current_layers[name]).__name__
                    params = layer_params.get(name, 0)
                else:
                    is_changed = (prev_params.get(name, 0) != layer_params.get(name, 0)) or name in changed_layers
                    prefix = "[CHANGED] " if is_changed else ""
                    layer_type = type(current_layers[name]).__name__
                    params = layer_params.get(name, 0)
            else:
                prefix = "[CHANGED] " if name in changed_layers else ""
                layer_type = type(current_layers[name]).__name__
                params = layer_params.get(name, 0)
            
            simplified_name = visualizer._simplify_layer_name(name)
            formatted_params = visualizer._format_param_count(params)
            
            print(f"{prefix}{simplified_name} {layer_type} ({formatted_params})")
    
    print("=" * 50)
    total_params = sum(layer_params.values())
    print(f"📊 Total: {visualizer._format_param_count(total_params)}")