"""
Device management utilities for NeuroExapt
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def move_module_to_device_like(new_module: nn.Module, reference_module: nn.Module) -> nn.Module:
    """
    Move a new module to the same device as a reference module.
    
    Args:
        new_module: The module to move
        reference_module: The reference module to get device from
        
    Returns:
        The new module moved to the appropriate device
    """
    try:
        # First try to get device from weight parameter
        if hasattr(reference_module, 'weight') and reference_module.weight is not None:
            device = reference_module.weight.device
            new_module = new_module.to(device)
            logger.debug(f"🔧 Module moved to device: {device} (via weight)")
            return new_module
        
        # Try to get device from any parameter
        if hasattr(reference_module, 'parameters'):
            try:
                device = next(reference_module.parameters()).device
                new_module = new_module.to(device)
                logger.debug(f"🔧 Module moved to device: {device} (via parameters)")
                return new_module
            except StopIteration:
                # No parameters in reference module
                logger.debug("🔧 Reference module has no parameters, keeping new module on current device")
                pass
        
        # If no parameters found, try to get device from buffers
        if hasattr(reference_module, 'buffers'):
            try:
                device = next(reference_module.buffers()).device
                new_module = new_module.to(device)
                logger.debug(f"🔧 Module moved to device: {device} (via buffers)")
                return new_module
            except StopIteration:
                # No buffers either
                pass
        
        # If all else fails, return the module as-is
        logger.debug("🔧 Could not determine reference device, keeping new module unchanged")
        return new_module
        
    except Exception as e:
        logger.warning(f"⚠️ Device transfer failed: {e}, keeping new module unchanged")
        return new_module


def get_module_device(module: nn.Module) -> torch.device:
    """
    Get the device of a module.
    
    Args:
        module: The module to check
        
    Returns:
        The device of the module, or CPU if no device can be determined
    """
    try:
        # Try weight first
        if hasattr(module, 'weight') and module.weight is not None:
            return module.weight.device
        
        # Try any parameter
        try:
            return next(module.parameters()).device
        except StopIteration:
            pass
        
        # Try any buffer
        try:
            return next(module.buffers()).device
        except StopIteration:
            pass
        
        # Default to CPU
        return torch.device('cpu')
        
    except Exception:
        return torch.device('cpu')


def ensure_same_device(*tensors_or_modules):
    """
    Ensure all tensors or modules are on the same device as the first one.
    
    Args:
        *tensors_or_modules: Variable number of tensors or modules
        
    Returns:
        List of tensors/modules moved to the same device
    """
    if not tensors_or_modules:
        return []
    
    # Get device from first item
    first_item = tensors_or_modules[0]
    if isinstance(first_item, nn.Module):
        target_device = get_module_device(first_item)
    else:
        target_device = first_item.device
    
    result = []
    for item in tensors_or_modules:
        if isinstance(item, nn.Module):
            # For modules, move entire module
            result.append(item.to(target_device))
        else:
            # For tensors
            result.append(item.to(target_device))
    
    return result