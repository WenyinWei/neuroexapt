"""
Unit tests for the main NeuroExapt class.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from neuroexapt import NeuroExapt
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleTestModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@unittest.skipIf(not IMPORTS_AVAILABLE, "Neuro Exapt components not available")
class TestNeuroExapt(unittest.TestCase):
    """Test cases for NeuroExapt main class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.model = SimpleTestModel()
        self.neuro_exapt = NeuroExapt(
            task_type="classification",
            device=self.device,
            verbose=False  # Disable verbose output in tests
        )
        
    def test_initialization(self):
        """Test NeuroExapt initialization."""
        self.assertIsNotNone(self.neuro_exapt)
        self.assertEqual(self.neuro_exapt.task_type, "classification")
        self.assertEqual(self.neuro_exapt.device, self.device)
        
    def test_model_wrapping(self):
        """Test model wrapping functionality."""
        wrapped_model = self.neuro_exapt.wrap_model(self.model)
        
        self.assertIsNotNone(wrapped_model)
        self.assertIsNotNone(self.neuro_exapt.model)
        self.assertIsNotNone(self.neuro_exapt.wrapped_model)
        
    def test_forward_pass(self):
        """Test forward pass through wrapped model."""
        wrapped_model = self.neuro_exapt.wrap_model(self.model)
        
        # Create dummy input
        batch_size = 4
        input_size = 10
        x = torch.randn(batch_size, input_size)
        
        # Forward pass
        output = wrapped_model(x)
        
        # Check output shape
        expected_shape = (batch_size, 5)  # output_size=5
        self.assertEqual(output.shape, expected_shape)
        
    def test_config_loading(self):
        """Test configuration loading."""
        # Default config should be loaded
        self.assertIsNotNone(self.neuro_exapt.config)
        self.assertIn('information', self.neuro_exapt.config)
        self.assertIn('entropy', self.neuro_exapt.config)
        self.assertIn('evolution', self.neuro_exapt.config)
        
    def test_state_saving_loading(self):
        """Test state saving and loading."""
        # Wrap model first
        self.neuro_exapt.wrap_model(self.model)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            save_path = tmp_file.name
            
        try:
            # Save state
            self.neuro_exapt.save_state(save_path)
            self.assertTrue(os.path.exists(save_path))
            
            # Create new instance and load state
            new_neuro_exapt = NeuroExapt(device=self.device, verbose=False)
            new_model = SimpleTestModel()
            new_neuro_exapt.wrap_model(new_model)
            new_neuro_exapt.load_state(save_path)
            
            # Check if states match
            self.assertEqual(new_neuro_exapt.current_epoch, self.neuro_exapt.current_epoch)
            
        finally:
            # Clean up
            if os.path.exists(save_path):
                os.unlink(save_path)
                
    def test_epoch_update(self):
        """Test epoch update functionality."""
        initial_epoch = self.neuro_exapt.current_epoch
        self.neuro_exapt.update_epoch(10, 100)
        
        self.assertEqual(self.neuro_exapt.current_epoch, 10)
        self.assertNotEqual(self.neuro_exapt.current_epoch, initial_epoch)
        
    def test_model_summary(self):
        """Test model summary generation."""
        # First wrap a model
        wrapped_model = self.neuro_exapt.wrap_model(self.model)
        
        # Test that wrapped model has the expected structure
        self.assertTrue(hasattr(wrapped_model, 'model'))
        self.assertTrue(hasattr(wrapped_model, 'forward'))
        self.assertTrue(hasattr(wrapped_model, 'compute_loss'))
        
    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test operations before model wrapping
        with self.assertRaises(ValueError):
            dummy_metrics = {'loss': 0.5, 'accuracy': 0.8}
            self.neuro_exapt.evolve_structure(dummy_metrics)
            
    def test_different_task_types(self):
        """Test initialization with different task types."""
        task_types = ["classification", "regression", "generation"]
        
        for task_type in task_types:
            ne = NeuroExapt(task_type=task_type, device=self.device, verbose=False)
            self.assertEqual(ne.task_type, task_type)
            
    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        self.assertIsNotNone(self.neuro_exapt.info_theory)
        self.assertIsNotNone(self.neuro_exapt.entropy_ctrl)
        self.assertIsNotNone(self.neuro_exapt.struct_optimizer)
        
        # Test operators
        self.assertIsNotNone(self.neuro_exapt.prune_op)
        self.assertIsNotNone(self.neuro_exapt.expand_op)
        self.assertIsNotNone(self.neuro_exapt.mutate_op)
        self.assertIsNotNone(self.neuro_exapt.compound_op)


class TestNeuroExaptWrapper(unittest.TestCase):
    """Test cases for NeuroExaptWrapper."""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Neuro Exapt components not available")
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = SimpleTestModel()
        self.neuro_exapt = NeuroExapt(device=self.device, verbose=False)
        
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Neuro Exapt components not available")
    def test_wrapper_forward(self):
        """Test wrapper forward pass."""
        wrapped_model = self.neuro_exapt.wrap_model(self.model)
        
        x = torch.randn(2, 10)
        output = wrapped_model(x)
        
        self.assertEqual(output.shape, (2, 5))
        
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Neuro Exapt components not available")
    def test_loss_computation(self):
        """Test combined loss computation."""
        wrapped_model = self.neuro_exapt.wrap_model(self.model)
        
        x = torch.randn(2, 10)
        targets = torch.randint(0, 5, (2,))
        
        predictions = wrapped_model(x)
        loss_fn = nn.CrossEntropyLoss()
        
        total_loss = wrapped_model.compute_loss(predictions, targets, loss_fn)
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(total_loss.requires_grad)


if __name__ == '__main__':
    unittest.main() 