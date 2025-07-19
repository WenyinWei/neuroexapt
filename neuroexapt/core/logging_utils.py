#!/usr/bin/env python3
"""
Logging Utilities for NeuroExapt - 统一日志系统

🔧 提供统一的日志接口，避免循环导入问题
"""

import logging
import os
import time
from typing import List


class ConfigurableLogger:
    """可配置的高性能日志系统，替代ANSI彩色打印"""
    
    def __init__(self, name: str = "neuroexapt", level: str = "INFO", enable_console: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 控制台处理器
            if enable_console:
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # 文件处理器（可选）
            log_file = os.environ.get('NEUROEXAPT_LOG_FILE')
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
        
        self.section_stack = []
        
    def debug(self, message: str, *args, **kwargs):
        """记录调试信息"""
        if self.logger.isEnabledFor(logging.DEBUG):
            indent = "  " * len(self.section_stack)
            self.logger.debug(f"{indent}{message}", *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """记录信息"""
        if self.logger.isEnabledFor(logging.INFO):
            indent = "  " * len(self.section_stack)
            self.logger.info(f"{indent}{message}", *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """记录警告"""
        if self.logger.isEnabledFor(logging.WARNING):
            indent = "  " * len(self.section_stack)
            self.logger.warning(f"{indent}{message}", *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """记录错误"""
        if self.logger.isEnabledFor(logging.ERROR):
            indent = "  " * len(self.section_stack)
            self.logger.error(f"{indent}{message}", *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """记录成功信息（使用INFO级别）"""
        if self.logger.isEnabledFor(logging.INFO):
            indent = "  " * len(self.section_stack)
            self.logger.info(f"{indent}✅ {message}", *args, **kwargs)
    
    def enter_section(self, section_name: str):
        """进入新的日志区域"""
        indent = "  " * len(self.section_stack)
        self.logger.info(f"{indent}🔍 进入 {section_name}")
        self.section_stack.append(section_name)
    
    def exit_section(self, section_name: str):
        """退出日志区域"""
        if self.section_stack and self.section_stack[-1] == section_name:
            self.section_stack.pop()
        indent = "  " * len(self.section_stack)
        self.logger.info(f"{indent}✅ 完成 {section_name}")
    
    def log_tensor_info(self, tensor, name: str):
        """记录张量信息"""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
            
        if tensor is None:
            self.warning(f"❌ {name}: None")
            return
        
        try:
            # 检查是否是torch tensor
            if hasattr(tensor, 'device') and hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
                device_info = f"({tensor.device})" if hasattr(tensor, 'device') else ""
                self.debug(f"📊 {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={device_info}")
            else:
                # 处理其他类型的tensor-like对象
                if hasattr(tensor, 'shape'):
                    self.debug(f"📊 {name}: shape={getattr(tensor, 'shape', 'unknown')}")
                else:
                    self.debug(f"📊 {name}: {type(tensor).__name__}")
        except Exception as e:
            self.warning(f"⚠️ 无法记录张量信息 {name}: {e}")
    
    def log_model_info(self, model, name: str = "Model"):
        """记录模型信息"""
        if not self.logger.isEnabledFor(logging.INFO):
            return
            
        try:
            # 检查是否是PyTorch模型
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # 尝试获取设备信息
                try:
                    device = next(model.parameters()).device if list(model.parameters()) else "Unknown"
                except StopIteration:
                    device = "Unknown"
                
                self.info(f"🧠 {name}: 总参数={total_params:,}, 可训练={trainable_params:,}, 设备={device}")
            else:
                # 处理其他类型的模型对象
                self.info(f"🧠 {name}: {type(model).__name__}")
        except Exception as e:
            self.warning(f"⚠️ 无法记录模型信息 {name}: {e}")


# 全局日志配置
_log_level = os.environ.get('NEUROEXAPT_LOG_LEVEL', 'INFO')
_enable_console = os.environ.get('NEUROEXAPT_ENABLE_CONSOLE', 'true').lower() in ('true', '1', 'yes')

# 创建全局logger实例
logger = ConfigurableLogger("neuroexapt.dnm", _log_level, _enable_console)


class DebugPrinter:
    """调试输出管理器 - 兼容旧版本接口"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.indent_level = 0
        
    def print_debug(self, message: str, level: str = "INFO"):
        """打印调试信息"""
        if not self.enabled:
            return
            
        indent = "  " * self.indent_level
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        # 使用统一的logger系统
        if level == "DEBUG":
            logger.debug(f"{message}")
        elif level == "INFO":
            logger.info(f"{message}")
        elif level == "WARNING":
            logger.warning(f"{message}")
        elif level == "ERROR":
            logger.error(f"{message}")
        elif level == "SUCCESS":
            logger.success(f"{message}")
        else:
            logger.info(f"{message}")
        
    def enter_section(self, section_name: str):
        """进入新的调试区域"""
        logger.enter_section(section_name)
        self.indent_level += 1
        
    def exit_section(self, section_name: str):
        """退出调试区域"""
        self.indent_level = max(0, self.indent_level - 1)
        logger.exit_section(section_name)
    
    def log_tensor_info(self, tensor, name: str):
        """记录张量信息（兼容接口）"""
        if self.enabled:
            logger.log_tensor_info(tensor, name)
    
    def log_model_info(self, model, name: str = "Model"):
        """记录模型信息（兼容接口）"""
        if self.enabled:
            logger.log_model_info(model, name)
    
    def print_tensor_info(self, tensor, name: str):
        """打印张量信息（兼容接口）"""
        if self.enabled:
            logger.log_tensor_info(tensor, name)
    
    def print_model_info(self, model, name: str = "Model"):
        """打印模型信息（兼容接口）"""
        if self.enabled:
            logger.log_model_info(model, name)


# Logger实例缓存，避免重复创建
_logger_cache = {}

def get_logger(name: str = None) -> ConfigurableLogger:
    """获取logger实例，支持缓存以保持状态一致性"""
    if name is None:
        return logger
    
    # 使用缓存避免重复创建logger实例
    if name not in _logger_cache:
        _logger_cache[name] = ConfigurableLogger(name, _log_level, _enable_console)
    
    return _logger_cache[name]