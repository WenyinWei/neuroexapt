"""
Quick test to verify information-guided evolution is working after the fix.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.neuroexapt import NeuroExapt


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.fc3 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    print("Testing information-guided evolution fix...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = SimpleNet()
    
    # Initialize NeuroExapt
    neuro_exapt = NeuroExapt(
        task_type="classification",
        device=device,
        verbose=True
    )
    
    # Wrap model
    wrapped_model = neuro_exapt.wrap_model(model)
    
    # Create sample data
    sample_data = torch.randn(16, 10).to(device)
    
    # Test evolution with information guidance
    try:
        performance_metrics = {
            'accuracy': 75.0,
            'val_accuracy': 70.0,
            'loss': 0.5
        }
        
        evolved, info = neuro_exapt.evolve_structure(
            performance_metrics,
            sample_data=sample_data
        )
        
        print("\n✅ Success! Information-guided evolution is working.")
        print(f"Evolution result: {evolved}")
        print(f"Action: {info.get('action', 'none')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 