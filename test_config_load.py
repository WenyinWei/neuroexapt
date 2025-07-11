"""
Test config loading for NeuroExapt
"""

import os
import yaml

# Check if config.yaml exists
config_path = os.path.join("neuroexapt", "config.yaml")
print(f"Config path: {config_path}")
print(f"Config exists: {os.path.exists(config_path)}")

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nLoaded config:")
    print(f"Evolution section: {config.get('evolution', {})}")
    print(f"Has expand_ratio: {'expand_ratio' in config.get('evolution', {})}")
    print(f"expand_ratio value: {config.get('evolution', {}).get('expand_ratio', 'NOT FOUND')}")
else:
    print("Config file not found!")

# Now test loading through NeuroExapt
print("\n" + "="*50)
print("Testing NeuroExapt initialization...")

from neuroexapt import NeuroExapt

# Create a minimal test
try:
    neuro = NeuroExapt(verbose=False)
    print("✅ NeuroExapt initialized successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Try to debug the config loading
    import neuroexapt.neuroexapt as ne
    test_ne = ne.NeuroExapt.__new__(ne.NeuroExapt)
    test_ne.device = "cpu"
    test_ne.config = test_ne._load_config(test_ne, None)
    print(f"\nDefault config evolution section: {test_ne.config.get('evolution', {})}")
    print(f"Has expand_ratio in defaults: {'expand_ratio' in test_ne.config.get('evolution', {})}") 