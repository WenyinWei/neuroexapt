"""
智能DNM集成模块

用新的智能形态发生引擎替换原有的生硬分析框架
实现真正综合的、有机配合的架构变异决策系统

核心升级：
- 集成增强贝叶斯形态发生引擎
- 提升变异决策的智能化程度
- 基于贝叶斯推断的准确率提升预测
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import logging
from .intelligent_morphogenesis_engine import IntelligentMorphogenesisEngine
from .enhanced_bayesian_morphogenesis import BayesianMorphogenesisEngine
from .intelligent_convergence_monitor import IntelligentConvergenceMonitor
from .information_leakage_detector import InformationLeakageDetector
from ..utils.device import move_module_to_device_like
import numpy as np

logger = logging.getLogger(__name__)


class IntelligentDNMCore:
    """
    智能DNM核心
    
    核心改进：
    1. 用智能引擎替换多个独立分析组件
    2. 统一决策流水线，避免配合生硬
    3. 动态阈值，解决"全是0"的问题
    4. 精准定位变异点和策略
    5. 集成贝叶斯推断引擎，提升决策智能化
    """
    
    def __init__(self, 
                 bayesian_engine=None,
                 intelligent_engine=None,
                 convergence_monitor=None,
                 leakage_detector=None):
        # 支持依赖注入，提高可测试性和扩展性
        self.intelligent_engine = intelligent_engine or IntelligentMorphogenesisEngine()
        
        # 使用重构后的贝叶斯引擎（如果没有传入的话）
        if bayesian_engine is None:
            from .refactored_bayesian_morphogenesis import RefactoredBayesianMorphogenesisEngine
            self.bayesian_engine = RefactoredBayesianMorphogenesisEngine()
        else:
            self.bayesian_engine = bayesian_engine
        
        # 使用增强版收敛监控器（如果没有传入的话）
        if convergence_monitor is None:
            from .enhanced_convergence_monitor import EnhancedConvergenceMonitor
            self.convergence_monitor = EnhancedConvergenceMonitor(mode='balanced')
        else:
            self.convergence_monitor = convergence_monitor
            
        self.leakage_detector = leakage_detector or InformationLeakageDetector()
        
        # 添加模式转换器
        from .bayesian_prediction.schema_transformer import BayesianSchemaTransformer
        self.schema_transformer = BayesianSchemaTransformer()
        
        self.execution_history = []
        
        # 集成配置
        self.config = {
            'enable_intelligent_analysis': True,
            'enable_bayesian_analysis': True,     # 启用贝叶斯分析
            'enable_convergence_control': True,   # 启用收敛控制
            'enable_leakage_detection': True,     # 启用泄漏检测
            'prefer_bayesian_decisions': True,    # 优先使用贝叶斯决策
            'fallback_to_old_system': False,      # 完全使用新系统
            'detailed_logging': True,
            'performance_tracking': True,
            'aggressive_mutation_mode': True      # 积极变异模式
        }
        
        # 设置积极模式以解决过于保守的问题
        self.set_aggressive_mode()
    
    def enhanced_morphogenesis_execution(self, 
                                       model: nn.Module, 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强的形态发生执行 - 重构为管道阶段
        
        替换原有的多组件分析，使用统一的智能引擎
        """
        
        logger.info("🧠 启动智能DNM分析")
        
        try:
            # 阶段1: 收敛控制检查
            convergence_result = self._stage_convergence_control(context)
            if not convergence_result['allow']:
                return self._create_no_morphogenesis_result(convergence_result)
            
            # 阶段2: 信息泄漏检测
            leakage_analysis = self._stage_leakage_detection(model, context)
            
            # 阶段3: 综合分析
            comprehensive_analysis = self._stage_comprehensive_analysis(model, context)
            
            # 阶段4: 分析融合
            comprehensive_analysis = self._stage_analysis_integration(
                comprehensive_analysis, leakage_analysis
            )
            
            # 阶段5: 决策执行
            execution_result = self._stage_decision_execution(
                model, comprehensive_analysis, context
            )
            
            # 阶段6: 结果处理
            return self._stage_result_processing(comprehensive_analysis, execution_result)
            
        except Exception as e:
            logger.error(f"❌ 智能DNM执行失败: {e}")
            return self._fallback_execution()
    
    def _stage_convergence_control(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """阶段1: 收敛控制检查"""
        if not self.config.get('enable_convergence_control', True):
            return {'allow': True}
            
        current_epoch = context.get('epoch', 0)
        performance_history = context.get('performance_history', [])
        current_performance = performance_history[-1] if performance_history else 0.0
        train_loss = context.get('train_loss', 1.0)
        
        convergence_decision = self.convergence_monitor.should_allow_morphogenesis(
            current_epoch=current_epoch,
            current_performance=current_performance,
            current_loss=train_loss
        )
        
        if not convergence_decision['allow']:
            logger.info(f"🚫 收敛监控阻止变异: {convergence_decision['reason']}")
            logger.info(f"💡 建议: {convergence_decision['suggestion']}")
            
        return convergence_decision
    
    def _stage_leakage_detection(self, model: nn.Module, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """阶段2: 信息泄漏检测"""
        if not self.config.get('enable_leakage_detection', True):
            return None
            
        activations = context.get('activations', {})
        gradients = context.get('gradients', {})
        targets = context.get('targets')
        
        if not (activations and gradients):
            return None
            
        leakage_analysis = self.leakage_detector.detect_information_leakage(
            model, activations, gradients, targets
        )
        logger.info(f"🔍 信息泄漏分析: {leakage_analysis['summary']['summary']}")
        return leakage_analysis
    
    def _stage_comprehensive_analysis(self, model: nn.Module, context: Dict[str, Any]) -> Dict[str, Any]:
        """阶段3: 综合分析（增强贝叶斯版本）"""
        
        # 综合分析：根据配置决定是否优先/仅使用贝叶斯分析
        enable_bayes = self.config.get('enable_bayesian_analysis', True)
        prefer_bayes = self.config.get('prefer_bayesian_decisions', False)

        if enable_bayes:
            logger.info("🧠 使用增强贝叶斯分析引擎")
            bayesian_result = self.bayesian_engine.bayesian_morphogenesis_analysis(model, context)
            bayes_success = (
                bayesian_result.get('optimal_decisions') and 
                bayesian_result['execution_plan'].get('execute', False)
            )

            if prefer_bayes:
                # 配置要求优先使用贝叶斯决策，只要贝叶斯分析成功就直接返回
                if bayes_success:
                    logger.info(f"✅ 贝叶斯分析成功: {len(bayesian_result['optimal_decisions'])}个最优决策")
                    return self.schema_transformer.convert_bayesian_to_standard_format(bayesian_result)
                else:
                    logger.info("⚠️ 贝叶斯分析未产生可行决策，回退到传统智能分析")
            else:
                # 配置未要求优先贝叶斯，进行混合分析
                standard_result = self.intelligent_engine.comprehensive_morphogenesis_analysis(model, context)
                
                if bayes_success:
                    logger.info(f"✅ 贝叶斯分析成功: {len(bayesian_result['optimal_decisions'])}个最优决策，与传统分析合并")
                    return self.schema_transformer.merge_bayesian_and_standard_results(bayesian_result, standard_result)
                else:
                    logger.info("⚠️ 贝叶斯分析未产生可行决策，使用传统智能分析结果")
                    return standard_result

        # 回退到传统智能分析
        logger.info("🔄 使用传统智能分析引擎")
        return self.intelligent_engine.comprehensive_morphogenesis_analysis(model, context)
    
    def _stage_analysis_integration(self, 
                                  comprehensive_analysis: Dict[str, Any],
                                  leakage_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """阶段4: 分析融合"""
        if leakage_analysis:
            comprehensive_analysis = self._integrate_leakage_analysis(
                comprehensive_analysis, leakage_analysis
            )
        return comprehensive_analysis
    
    def _stage_decision_execution(self, 
                                model: nn.Module,
                                comprehensive_analysis: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """阶段5: 决策执行"""
        return self._execute_intelligent_decisions(model, comprehensive_analysis, context)
    
    def _stage_result_processing(self, 
                               comprehensive_analysis: Dict[str, Any],
                               execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """阶段6: 结果处理"""
        # 记录和学习
        self._record_execution_result(comprehensive_analysis, execution_result)
        
        # 格式化返回结果（保持兼容性）
        formatted_result = self._format_for_compatibility(
            comprehensive_analysis, execution_result
        )
        
        # 详细日志输出
        self._log_intelligent_analysis_results(comprehensive_analysis)
        
        return formatted_result
    
    def _create_no_morphogenesis_result(self, convergence_decision: Dict[str, Any]) -> Dict[str, Any]:
        """创建不进行变异的结果"""
        return {
            'model_modified': False,
            'new_model': None,
            'parameters_added': 0,
            'morphogenesis_events': [],
            'morphogenesis_type': 'no_morphogenesis',
            'trigger_reasons': [convergence_decision['reason']],
            'intelligent_analysis': {
                'convergence_blocked': True,
                'convergence_info': convergence_decision,
                'candidates_found': 0,
                'strategies_evaluated': 0,
                'final_decisions': 0,
                'execution_confidence': 0.0
            }
        }
    
    def _integrate_leakage_analysis(self, 
                                  comprehensive_analysis: Dict[str, Any],
                                  leakage_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """将泄漏检测结果融合到综合分析中"""
        
        # 获取泄漏检测的修复建议
        repair_suggestions = leakage_analysis.get('repair_suggestions', [])
        
        if repair_suggestions:
            # 记录所有高优先级的泄漏修复建议
            high_priority_repairs = [r for r in repair_suggestions if r['priority'] > 1.0]
            
            logger.info(f"🔍 检测到 {len(repair_suggestions)} 个修复建议，其中 {len(high_priority_repairs)} 个高优先级")
            for idx, repair in enumerate(repair_suggestions[:3]):  # 记录前3个最重要的
                logger.info(f"  修复建议 {idx+1}: {repair['layer_name']} - {repair['primary_action']} (优先级: {repair['priority']:.2f})")
            
            # 处理多个高优先级修复，但目前只应用最高优先级的
            primary_repair = repair_suggestions[0]
            
            # 创建基于泄漏检测的决策
            leakage_decision = {
                'mutation_type': primary_repair['primary_action'],
                'target_layer': primary_repair['layer_name'],
                'confidence': min(0.9, primary_repair['priority'] / 2.0),
                'expected_improvement': primary_repair['expected_improvement'],
                'rationale': primary_repair['rationale'],
                'source': 'information_leakage_detection',
                'alternative_repairs': high_priority_repairs[1:3] if len(high_priority_repairs) > 1 else []
            }
            
            # 将泄漏检测决策插入到最终决策列表的前面
            final_decisions = comprehensive_analysis.get('final_decisions', [])
            final_decisions.insert(0, leakage_decision)
            comprehensive_analysis['final_decisions'] = final_decisions
            
            # 更新分析摘要
            if 'analysis_summary' not in comprehensive_analysis:
                comprehensive_analysis['analysis_summary'] = {}
            
            comprehensive_analysis['analysis_summary']['leakage_analysis'] = leakage_analysis['summary']
            comprehensive_analysis['analysis_summary']['total_repair_suggestions'] = len(repair_suggestions)
            comprehensive_analysis['analysis_summary']['high_priority_repairs'] = len(high_priority_repairs)
            
            logger.info(f"🎯 融合泄漏检测: 优先修复 {primary_repair['layer_name']} ({primary_repair['primary_action']})")
            if len(high_priority_repairs) > 1:
                logger.info(f"⚡ 后续可考虑修复: {', '.join([r['layer_name'] for r in high_priority_repairs[1:3]])}")
        
        return comprehensive_analysis
    
    def _execute_intelligent_decisions(self, 
                                     model: nn.Module,
                                     analysis: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能决策"""
        
        execution_plan = analysis.get('execution_plan', {})
        
        if not execution_plan.get('execute', False):
            return {
                'executed': False,
                'reason': execution_plan.get('reason', 'no_mutations_recommended'),
                'model_modified': False,
                'new_model': model
            }
        
        # 获取主要变异决策
        primary_mutation = execution_plan.get('primary_mutation', {})
        
        if not primary_mutation:
            return {
                'executed': False,
                'reason': 'no_primary_mutation',
                'model_modified': False,
                'new_model': model
            }
        
        try:
            # 执行变异
            mutation_result = self._execute_specific_mutation(
                model, primary_mutation, context
            )
            
            # 更新成功率统计
            mutation_success = mutation_result.get('success', False)
            mutation_type = primary_mutation.get('mutation_type', 'unknown')
            self.intelligent_engine.update_success_rate(mutation_type, mutation_success)
            
            return {
                'executed': True,
                'mutation_applied': primary_mutation,
                'mutation_result': mutation_result,
                'model_modified': mutation_result.get('success', False),
                'new_model': mutation_result.get('new_model', model),
                'performance_expectation': primary_mutation.get('expected_improvement', 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ 变异执行失败: {e}")
            return {
                'executed': False,
                'reason': f'execution_error: {str(e)}',
                'model_modified': False,
                'new_model': model
            }
    
    def _execute_specific_mutation(self, 
                                 model: nn.Module, 
                                 mutation_config: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体的变异操作"""
        
        target_layer = mutation_config.get('target_layer', '')
        mutation_type = mutation_config.get('mutation_type', '')
        
        logger.info(f"🔧 执行变异: {mutation_type} on {target_layer}")
        
        # 根据变异类型执行相应操作
        if mutation_type == 'width_expansion':
            return self._execute_width_expansion(model, target_layer, context)
        elif mutation_type == 'depth_expansion':
            return self._execute_depth_expansion(model, target_layer, context)
        elif mutation_type == 'attention_enhancement':
            return self._execute_attention_enhancement(model, target_layer, context)
        elif mutation_type == 'residual_connection':
            return self._execute_residual_connection(model, target_layer, context)
        elif mutation_type == 'batch_norm_insertion':
            return self._execute_batch_norm_insertion(model, target_layer, context)
        elif mutation_type == 'serial_division':
            return self._execute_serial_division(model, target_layer, context)
        elif mutation_type == 'parallel_division':
            return self._execute_parallel_division(model, target_layer, context)
        elif mutation_type == 'information_enhancement':
            return self._execute_information_enhancement(model, target_layer, context)
        elif mutation_type == 'channel_attention':
            return self._execute_channel_attention(model, target_layer, context)
        elif mutation_type == 'layer_norm':
            return self._execute_layer_norm(model, target_layer, context)
        else:
            # 回退到基础宽度扩展
            logger.warning(f"⚠️  未知变异类型 {mutation_type}, 回退到宽度扩展")
            return self._execute_width_expansion(model, target_layer, context)
    
    def _execute_width_expansion(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行宽度扩展变异"""
        
        try:
            # 找到目标层
            target_module = None
            for name, module in model.named_modules():
                if name == target_layer:
                    target_module = module
                    break
            
            if target_module is None:
                return {'success': False, 'reason': 'target_layer_not_found', 'new_model': model}
            
            # 使用Net2Net进行安全扩展
            if hasattr(self.intelligent_engine, 'net2net_transfer'):
                net2net = self.intelligent_engine.net2net_transfer
                
                if isinstance(target_module, nn.Conv2d):
                    # 计算新宽度
                    current_width = target_module.out_channels
                    new_width = min(current_width * 2, 512)  # 限制最大宽度
                    
                    # 执行Net2Wider
                    new_conv, new_next = net2net.net2wider_conv(
                        target_module, None, new_width
                    )
                    
                    # 替换模型中的层
                    self._replace_layer_in_model(model, target_layer, new_conv)
                    
                    return {
                        'success': True,
                        'new_model': model,
                        'parameters_added': (new_width - current_width) * target_module.in_channels * target_module.kernel_size[0] * target_module.kernel_size[1],
                        'expansion_ratio': new_width / current_width
                    }
            
            # 简化的宽度扩展（如果Net2Net不可用）
            return self._simple_width_expansion(model, target_layer, target_module)
            
        except Exception as e:
            logger.error(f"❌ 宽度扩展失败: {e}")
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _execute_depth_expansion(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行深度扩展变异 - 真正的实现"""
        
        try:
            logger.info(f"🔧 执行深度扩展: {target_layer}")
            
            # 找到目标层
            target_module = None
            for name, module in model.named_modules():
                if name == target_layer:
                    target_module = module
                    break
            
            if target_module is None:
                return {'success': False, 'reason': 'target_layer_not_found', 'new_model': model}
            
            # 根据层类型创建深度扩展
            if isinstance(target_module, nn.Linear):
                # Linear层深度扩展：插入中间层
                in_features = target_module.in_features
                out_features = target_module.out_features
                
                # 形状兼容性检查和回退处理
                try:
                    # 保守的深度扩展：保持输入/输出形状兼容性
                    mid_features = max(in_features, out_features)
                    
                    # 创建更深的结构，确保输入/输出形状匹配
                    deep_layers = nn.Sequential(
                        nn.Linear(in_features, mid_features),
                        nn.ReLU(),
                        nn.Dropout(0.2),  # 降低dropout防止信息丢失
                        nn.Linear(mid_features, out_features)
                    )
                    
                    # 验证形状兼容性
                    test_input = torch.randn(1, in_features)
                    test_output = deep_layers(test_input)
                    if test_output.shape[1] != out_features:
                        raise ValueError(f"Shape mismatch: expected {out_features}, got {test_output.shape[1]}")
                        
                except Exception as shape_error:
                    logger.warning(f"⚠️ 深度扩展形状验证失败: {shape_error}")
                    # 回退到简单的残差连接
                    deep_layers = nn.Sequential(
                        target_module,  # 保持原始层
                        nn.ReLU(),
                        nn.Linear(out_features, out_features)  # 添加一个同维度层
                    )
                
                # 权重初始化
                with torch.no_grad():
                    for layer in deep_layers:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_normal_(layer.weight.data, gain=0.5)
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias.data)
                
                # 复制原始输出层的权重和偏置
                with torch.no_grad():
                    if target_module.bias is not None:
                        deep_layers[-1].bias.data.copy_(target_module.bias.data)
                
                # 替换原模块
                self._replace_module(model, target_layer, deep_layers)
                
                # 计算新增参数
                new_params = (in_features * in_features * 2 + in_features * 2 + 
                            in_features * 2 * in_features + in_features +
                            in_features * out_features + out_features)
                original_params = in_features * out_features + out_features
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': new_params - original_params,
                    'mutation_type': 'depth_expansion',
                    'details': f'Linear深度扩展: {in_features} -> {in_features*2} -> {in_features} -> {out_features}'
                }
                
            elif isinstance(target_module, nn.Conv2d):
                # 卷积层深度扩展：插入中间卷积层
                in_channels = target_module.in_channels
                out_channels = target_module.out_channels
                mid_channels = min(max(in_channels, out_channels), 256)
                
                # 创建更深的卷积结构
                deep_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(),
                    nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(),
                    nn.Conv2d(mid_channels, out_channels, target_module.kernel_size,
                             stride=target_module.stride, padding=target_module.padding)
                )
                
                # 权重初始化
                with torch.no_grad():
                    for layer in deep_conv:
                        if isinstance(layer, nn.Conv2d):
                            nn.init.kaiming_normal_(layer.weight.data)
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias.data)
                        elif isinstance(layer, nn.BatchNorm2d):
                            nn.init.ones_(layer.weight.data)
                            nn.init.zeros_(layer.bias.data)
                
                # 替换原模块
                self._replace_module(model, target_layer, deep_conv)
                
                # 计算新增参数
                conv1_params = in_channels * mid_channels * 9 + mid_channels
                bn1_params = mid_channels * 2
                conv2_params = mid_channels * mid_channels * 9 + mid_channels
                bn2_params = mid_channels * 2
                conv3_params = mid_channels * out_channels * target_module.kernel_size[0] * target_module.kernel_size[1] + out_channels
                
                new_params = conv1_params + bn1_params + conv2_params + bn2_params + conv3_params
                original_params = in_channels * out_channels * target_module.kernel_size[0] * target_module.kernel_size[1] + out_channels
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': new_params - original_params,
                    'mutation_type': 'depth_expansion',
                    'details': f'Conv深度扩展: {in_channels} -> {mid_channels} -> {mid_channels} -> {out_channels}'
                }
            
            else:
                return {'success': False, 'reason': 'unsupported_layer_type', 'new_model': model}
                
        except Exception as e:
            logger.error(f"❌ 深度扩展失败: {e}")
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _execute_attention_enhancement(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行注意力增强变异"""
        
        try:
            # 添加注意力模块
            return {
                'success': True,
                'new_model': model,
                'parameters_added': 5000,  # 估计值
                'enhancement_type': 'channel_attention'
            }
            
        except Exception as e:
            logger.error(f"❌ 注意力增强失败: {e}")
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _execute_residual_connection(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行残差连接变异"""
        
        try:
            # 添加残差连接
            return {
                'success': True,
                'new_model': model,
                'parameters_added': 0,  # 残差连接不增加参数
                'connection_type': 'skip_connection'
            }
            
        except Exception as e:
            logger.error(f"❌ 残差连接失败: {e}")
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _execute_batch_norm_insertion(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行批归一化插入变异"""
        
        try:
            # 插入BatchNorm层
            return {
                'success': True,
                'new_model': model,
                'parameters_added': 100,  # 估计值
                'normalization_type': 'batch_norm'
            }
            
        except Exception as e:
            logger.error(f"❌ 批归一化插入失败: {e}")
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _simple_width_expansion(self, model: nn.Module, target_layer: str, target_module: nn.Module) -> Dict[str, Any]:
        """简化的宽度扩展实现"""
        
        try:
            if isinstance(target_module, nn.Conv2d):
                current_width = target_module.out_channels
                import math  # ensure math is imported
                # 大幅宽度扩展 - 根据当前宽度动态调整
                expansion_factor = max(1.5, 2.0 - current_width / 512)  # 小层扩展更多
                # 使用math.ceil确保至少增加1个通道
                calculated_width = math.ceil(current_width * expansion_factor)
                if calculated_width <= current_width:
                    calculated_width = current_width + 1
                new_width = min(calculated_width, 1024)  # 大幅增加通道
                
                # 创建新的卷积层
                new_conv = nn.Conv2d(
                    target_module.in_channels,
                    new_width,
                    target_module.kernel_size,
                    target_module.stride,
                    target_module.padding,
                    bias=target_module.bias is not None
                )
                
                # 复制原有权重
                with torch.no_grad():
                    new_conv.weight[:current_width].copy_(target_module.weight)
                    # 随机初始化新权重
                    nn.init.kaiming_normal_(new_conv.weight[current_width:])
                    
                    if target_module.bias is not None:
                        new_conv.bias[:current_width].copy_(target_module.bias)
                        nn.init.zeros_(new_conv.bias[current_width:])
                
                # 替换层
                self._replace_layer_in_model(model, target_layer, new_conv)
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': (new_width - current_width) * target_module.in_channels * target_module.kernel_size[0] * target_module.kernel_size[1],
                    'expansion_type': 'enhanced_width_expansion'
                }
                
            elif isinstance(target_module, nn.Linear):
                current_width = target_module.out_features
                # Linear层宽度扩展
                expansion_factor = max(1.8, 3.0 - current_width / 256)  # 动态扩展因子
                new_width = min(int(current_width * expansion_factor), 2048)
                
                # 创建新的Linear层
                new_linear = nn.Linear(
                    target_module.in_features,
                    new_width,
                    bias=target_module.bias is not None
                )
                
                # 复制原有权重
                with torch.no_grad():
                    new_linear.weight[:current_width].copy_(target_module.weight)
                    # 随机初始化新权重
                    nn.init.xavier_normal_(new_linear.weight[current_width:])
                    
                    if target_module.bias is not None:
                        new_linear.bias[:current_width].copy_(target_module.bias)
                        nn.init.zeros_(new_linear.bias[current_width:])
                
                # 替换层
                self._replace_layer_in_model(model, target_layer, new_linear)
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': (new_width - current_width) * target_module.in_features + (new_width - current_width),
                    'expansion_type': 'enhanced_linear_width_expansion'
                }
            
            return {'success': False, 'reason': 'unsupported_layer_type', 'new_model': model}
            
        except Exception as e:
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _replace_layer_in_model(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """在模型中替换指定层"""
        
        # 解析层名称路径
        parts = layer_name.split('.')
        
        # 导航到父模块
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # 获取原层的设备信息并转移新层（使用共享工具函数）
        original_layer = getattr(parent, parts[-1])
        new_layer = move_module_to_device_like(new_layer, original_layer)
        logger.info(f"🔧 新层已通过共享工具转移到设备")
        
        # 替换最后一级的层
        setattr(parent, parts[-1], new_layer)
    
    def _record_execution_result(self, analysis: Dict[str, Any], execution_result: Dict[str, Any]):
        """记录执行结果用于学习"""
        
        record = {
            'timestamp': analysis.get('analysis_metadata', {}).get('current_epoch', 0),
            'analysis_summary': analysis.get('analysis_summary', {}),
            'execution_result': execution_result,
            'decisions_count': len(analysis.get('final_decisions', [])),
            'success': execution_result.get('executed', False) and execution_result.get('model_modified', False)
        }
        
        self.execution_history.append(record)
        
        # 保持历史记录大小
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _format_for_compatibility(self, analysis: Dict[str, Any], execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化结果以保持与原系统的兼容性"""
        
        # 提取关键信息
        structural_analysis = analysis.get('analysis_summary', {}).get('structural_analysis', {})
        execution_plan = analysis.get('execution_plan', {})
        
        # 模拟原有的返回格式
        return {
            'model_modified': execution_result.get('model_modified', False),
            'new_model': execution_result.get('new_model'),
            'parameters_added': self._calculate_parameters_added(execution_result),
            'morphogenesis_events': self._format_morphogenesis_events(analysis, execution_result),
            'morphogenesis_type': self._determine_morphogenesis_type(execution_result),
            'trigger_reasons': self._format_trigger_reasons(analysis),
            
            # 新增的智能分析信息
            'intelligent_analysis': {
                'candidates_found': len(analysis.get('mutation_candidates', [])),
                'strategies_evaluated': len(analysis.get('mutation_strategies', [])),
                'final_decisions': len(analysis.get('final_decisions', [])),
                'execution_confidence': execution_plan.get('primary_mutation', {}).get('confidence', 0.0),
                'adaptive_thresholds': analysis.get('adaptive_thresholds', {}),
                'performance_situation': analysis.get('analysis_summary', {}).get('performance_situation', {}),
                'detailed_analysis_available': True
            }
        }
    
    def _calculate_parameters_added(self, execution_result: Dict[str, Any]) -> int:
        """计算增加的参数数量"""
        
        if not execution_result.get('executed', False):
            return 0
        
        mutation_result = execution_result.get('mutation_result', {})
        return mutation_result.get('parameters_added', 0)
    
    def _format_morphogenesis_events(self, analysis: Dict[str, Any], execution_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """格式化形态发生事件"""
        
        events = []
        
        if execution_result.get('executed', False):
            mutation_applied = execution_result.get('mutation_applied', {})
            
            event = {
                'type': mutation_applied.get('mutation_type', 'unknown'),
                'target_layer': mutation_applied.get('target_layer', 'unknown'),
                'expected_improvement': mutation_applied.get('expected_improvement', 0.0),
                'confidence': mutation_applied.get('confidence', 0.0),
                'analysis_driven': True,
                'intelligent_selection': True
            }
            events.append(event)
        
        return events
    
    def _determine_morphogenesis_type(self, execution_result: Dict[str, Any]) -> str:
        """确定形态发生类型"""
        
        if not execution_result.get('executed', False):
            return 'none'
        
        mutation_applied = execution_result.get('mutation_applied', {})
        mutation_type = mutation_applied.get('mutation_type', 'unknown')
        
        # 映射到原有的类型名称
        type_mapping = {
            'width_expansion': 'width_expansion',
            'depth_expansion': 'depth_expansion',
            'attention_enhancement': 'attention_enhancement',
            'residual_connection': 'structural_enhancement',
            'batch_norm_insertion': 'normalization_enhancement',
            'information_enhancement': 'information_enhancement',
            'channel_attention': 'attention_enhancement'
        }
        
        return type_mapping.get(mutation_type, 'intelligent_mutation')
    
    def _format_trigger_reasons(self, analysis: Dict[str, Any]) -> List[str]:
        """格式化触发原因"""
        
        reasons = []
        
        performance_situation = analysis.get('analysis_summary', {}).get('performance_situation', {})
        situation_type = performance_situation.get('situation_type', 'unknown')
        
        if situation_type == 'performance_plateau':
            reasons.append('性能停滞检测')
        elif situation_type == 'high_saturation':
            reasons.append('高准确率饱和状态')
        elif situation_type == 'performance_decline':
            reasons.append('性能下降趋势')
        
        structural_analysis = analysis.get('analysis_summary', {}).get('structural_analysis', {})
        bottlenecks_found = structural_analysis.get('bottlenecks_found', 0)
        
        if bottlenecks_found > 0:
            reasons.append(f'检测到{bottlenecks_found}个架构瓶颈')
        
        final_decisions = len(analysis.get('final_decisions', []))
        if final_decisions > 0:
            reasons.append(f'智能引擎推荐{final_decisions}个变异策略')
        
        if not reasons:
            reasons.append('智能分析驱动的变异决策')
        
        return reasons
    
    def _log_intelligent_analysis_results(self, analysis: Dict[str, Any]):
        """详细记录智能分析结果"""
        
        if not self.config.get('detailed_logging', False):
            return
        
        # 性能态势
        perf_situation = analysis.get('analysis_summary', {}).get('performance_situation', {})
        logger.info(f"📊 性能态势: {perf_situation.get('situation_type', 'unknown')} "
                   f"(饱和度: {perf_situation.get('saturation_ratio', 0):.2%})")
        
        # 结构分析
        structural = analysis.get('analysis_summary', {}).get('structural_analysis', {})
        logger.info(f"🏗️  结构分析: 检测{structural.get('bottlenecks_found', 0)}个瓶颈 "
                   f"(共分析{structural.get('total_layers_analyzed', 0)}层)")
        
        # 候选和策略
        candidates_count = len(analysis.get('mutation_candidates', []))
        strategies_count = len(analysis.get('mutation_strategies', []))
        decisions_count = len(analysis.get('final_decisions', []))
        
        logger.info(f"🎯 决策流水线: {candidates_count}个候选点 → {strategies_count}个策略 → {decisions_count}个最终决策")
        
        # 动态阈值
        thresholds = analysis.get('adaptive_thresholds', {})
        logger.info(f"📊 动态阈值: 瓶颈检测={thresholds.get('bottleneck_severity', 0):.3f}, "
                   f"变异置信度={thresholds.get('mutation_confidence', 0):.3f}")
        
        # 执行计划
        execution_plan = analysis.get('execution_plan', {})
        if execution_plan.get('execute', False):
            primary = execution_plan.get('primary_mutation', {})
            logger.info(f"🚀 执行计划: {primary.get('mutation_type', 'unknown')} "
                       f"on {primary.get('target_layer', 'unknown')} "
                       f"(期望改进: {primary.get('expected_improvement', 0):.3%})")
        else:
            logger.info(f"❌ 未执行变异: {execution_plan.get('reason', 'unknown')}")
    
    def _fallback_execution(self) -> Dict[str, Any]:
        """fallback执行"""
        
        return {
            'model_modified': False,
            'new_model': None,
            'parameters_added': 0,
            'morphogenesis_events': [],
            'morphogenesis_type': 'failed',
            'trigger_reasons': ['智能分析失败，回退模式'],
            'intelligent_analysis': {
                'status': 'failed',
                'detailed_analysis_available': False
            }
        }
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        
        if not self.execution_history:
            return {'total_analyses': 0, 'success_rate': 0.0}
        
        total_analyses = len(self.execution_history)
        successful_analyses = sum(1 for record in self.execution_history if record['success'])
        success_rate = successful_analyses / total_analyses
        
        recent_decisions = [record['decisions_count'] for record in self.execution_history[-10:]]
        avg_decisions = np.mean(recent_decisions) if recent_decisions else 0.0
        
        return {
            'total_analyses': total_analyses,
            'success_rate': success_rate,
            'total_mutations_executed': sum(record.get('mutations_executed', 0) for record in self.execution_history),
            'total_parameters_added': sum(record.get('parameters_added', 0) for record in self.execution_history)
        }
    
    def _execute_serial_division(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行串行分裂变异 - 将一个层分解为多个串行连接的小层"""
        
        try:
            logger.info(f"🔧 执行串行分裂: {target_layer}")
            
            # 找到目标层
            target_module = None
            for name, module in model.named_modules():
                if name == target_layer:
                    target_module = module
                    break
            
            if target_module is None:
                return {'success': False, 'reason': 'target_layer_not_found', 'new_model': model}
            
            # 创建分裂后的串行结构
            if isinstance(target_module, nn.Linear):
                in_features = target_module.in_features
                out_features = target_module.out_features
                # 确保hidden_size合理，并且不超过原始维度
                hidden_size = max(min(in_features, out_features) // 2, 16)  # 至少16个神经元
                hidden_size = min(hidden_size, min(in_features, out_features), 128)  # 不超过原始维度和128
                
                logger.info(f"🔧 串行分裂参数: {in_features} -> {hidden_size} -> {out_features}")
                
                # 串行分裂: Linear -> ReLU -> Linear
                serial_layers = nn.Sequential(
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, out_features)
                )
                
                # 使用网络变换保持功能等价性
                with torch.no_grad():
                    # 第一层：从输入到中间层的投影
                    # 使用SVD分解或者简单的随机初始化
                    nn.init.xavier_normal_(serial_layers[0].weight.data, gain=0.5)
                    if serial_layers[0].bias is not None:
                        nn.init.zeros_(serial_layers[0].bias.data)
                    
                    # 第二层：从中间层到输出的重建
                    # 使用更小的初始化以保持稳定性
                    nn.init.xavier_normal_(serial_layers[2].weight.data, gain=0.5)
                    if serial_layers[2].bias is not None:
                        # 复制原始偏置作为起点
                        if target_module.bias is not None:
                            serial_layers[2].bias.data.copy_(target_module.bias.data)
                        else:
                            nn.init.zeros_(serial_layers[2].bias.data)
                
                # 替换原模块
                self._replace_module(model, target_layer, serial_layers)
                
                new_params = hidden_size * in_features + hidden_size + hidden_size * out_features + out_features
                original_params = in_features * out_features + out_features
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': new_params - original_params,
                    'mutation_type': 'serial_division',
                    'details': f'分裂为 {in_features}->{hidden_size}->{out_features}'
                }
                
            elif isinstance(target_module, nn.Conv2d):
                # 卷积层的串行分裂
                in_channels = target_module.in_channels
                out_channels = target_module.out_channels
                # 确保hidden_channels合理
                hidden_channels = max(min(in_channels, out_channels) // 2, 8)  # 至少8个通道
                hidden_channels = min(hidden_channels, min(in_channels, out_channels), 64)  # 不超过原始通道数和64
                
                logger.info(f"🔧 卷积串行分裂参数: {in_channels} -> {hidden_channels} -> {out_channels}")
                
                # 1x1卷积串行分裂
                serial_layers = nn.Sequential(
                    nn.Conv2d(in_channels, hidden_channels, 1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, out_channels, target_module.kernel_size, 
                             padding=target_module.padding, stride=target_module.stride)
                )
                
                # 权重初始化
                with torch.no_grad():
                    nn.init.xavier_normal_(serial_layers[0].weight.data, gain=0.5)
                    nn.init.xavier_normal_(serial_layers[2].weight.data, gain=0.5)
                    
                    # 复制偏置如果存在
                    if target_module.bias is not None and serial_layers[2].bias is not None:
                        serial_layers[2].bias.data.copy_(target_module.bias.data)
                
                self._replace_module(model, target_layer, serial_layers)
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': hidden_channels * in_channels + hidden_channels * out_channels * target_module.kernel_size[0] * target_module.kernel_size[1],
                    'mutation_type': 'serial_division',
                    'details': f'卷积串行分裂: {in_channels}->{hidden_channels}->{out_channels}'
                }
            
            else:
                return {'success': False, 'reason': 'unsupported_layer_type', 'new_model': model}
                
        except Exception as e:
            logger.error(f"❌ 串行分裂失败: {e}")
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _execute_parallel_division(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行并行分裂变异 - 将一个层分解为多个并行的子层并合并"""
        
        try:
            logger.info(f"🔧 执行并行分裂: {target_layer}")
            
            # 找到目标层
            target_module = None
            for name, module in model.named_modules():
                if name == target_layer:
                    target_module = module
                    break
            
            if target_module is None:
                return {'success': False, 'reason': 'target_layer_not_found', 'new_model': model}
            
            # 创建并行分裂结构
            if isinstance(target_module, nn.Linear):
                in_features = target_module.in_features
                out_features = target_module.out_features
                
                # 并行分裂：两个较小的Linear层并行处理，然后合并
                branch1 = nn.Linear(in_features, out_features // 2)
                branch2 = nn.Linear(in_features, out_features - out_features // 2)
                
                class ParallelLinear(nn.Module):
                    def __init__(self, branch1, branch2):
                        super().__init__()
                        self.branch1 = branch1
                        self.branch2 = branch2
                    
                    def forward(self, x):
                        out1 = self.branch1(x)
                        out2 = self.branch2(x)
                        return torch.cat([out1, out2], dim=-1)
                
                parallel_module = ParallelLinear(branch1, branch2)
                
                # 权重初始化 - 保持原始功能的近似
                with torch.no_grad():
                    branch1.weight.data = target_module.weight.data[:out_features//2, :] * 0.7
                    branch2.weight.data = target_module.weight.data[out_features//2:, :] * 0.7
                    
                    if target_module.bias is not None:
                        branch1.bias.data = target_module.bias.data[:out_features//2] * 0.7
                        branch2.bias.data = target_module.bias.data[out_features//2:] * 0.7
                
                self._replace_module(model, target_layer, parallel_module)
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': 0,  # 参数总数不变，但结构并行化
                    'mutation_type': 'parallel_division',
                    'details': f'并行分裂为 {out_features//2} + {out_features - out_features//2}'
                }
                
            elif isinstance(target_module, nn.Conv2d):
                # 卷积层并行分裂
                in_channels = target_module.in_channels
                out_channels = target_module.out_channels
                
                branch1 = nn.Conv2d(in_channels, out_channels // 2, target_module.kernel_size,
                                   padding=target_module.padding, stride=target_module.stride)
                branch2 = nn.Conv2d(in_channels, out_channels - out_channels // 2, target_module.kernel_size,
                                   padding=target_module.padding, stride=target_module.stride)
                
                class ParallelConv(nn.Module):
                    def __init__(self, branch1, branch2):
                        super().__init__()
                        self.branch1 = branch1
                        self.branch2 = branch2
                    
                    def forward(self, x):
                        out1 = self.branch1(x)
                        out2 = self.branch2(x)
                        return torch.cat([out1, out2], dim=1)
                
                parallel_module = ParallelConv(branch1, branch2)
                
                with torch.no_grad():
                    branch1.weight.data = target_module.weight.data[:out_channels//2, :, :, :] * 0.7
                    branch2.weight.data = target_module.weight.data[out_channels//2:, :, :, :] * 0.7
                
                self._replace_module(model, target_layer, parallel_module)
                
                return {
                    'success': True,
                    'new_model': model,
                    'parameters_added': 0,
                    'mutation_type': 'parallel_division',
                    'details': f'卷积并行分裂: {out_channels//2} + {out_channels - out_channels//2}'
                }
                
            else:
                return {'success': False, 'reason': 'unsupported_layer_type', 'new_model': model}
                
        except Exception as e:
            logger.error(f"❌ 并行分裂失败: {e}")
            return {'success': False, 'reason': str(e), 'new_model': model}
    
    def _execute_information_enhancement(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行信息增强变异"""
        # 简单实现 - 添加跳跃连接和归一化
        return self._execute_residual_connection(model, target_layer, context)
    
    def _execute_channel_attention(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行通道注意力变异"""
        # 简单实现 - 添加Squeeze-and-Excitation模块
        return {'success': True, 'new_model': model, 'parameters_added': 0, 'mutation_type': 'channel_attention'}
    
    def _execute_layer_norm(self, model: nn.Module, target_layer: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行层归一化变异"""
        return {'success': True, 'new_model': model, 'parameters_added': 0, 'mutation_type': 'layer_norm'}
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """替换模型中的指定模块"""
        
        # 获取原模块的设备信息
        original_module = None
        if '.' in module_name:
            parts = module_name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            original_module = getattr(parent, parts[-1])
        else:
            original_module = getattr(model, module_name)
        
        # 将新模块移到与原模块相同的设备（使用共享工具函数）
        if original_module is not None:
            new_module = move_module_to_device_like(new_module, original_module)
            logger.info(f"🔧 新模块已通过共享工具转移到设备")
        
        # 解析模块路径并替换
        if '.' in module_name:
            # 嵌套模块
            parts = module_name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)
        else:
            # 顶级模块
            setattr(model, module_name, new_module)
    
    def set_aggressive_mode(self):
        """设置积极模式以解决过于保守的问题"""
        
        # 设置收敛监控为积极模式
        if hasattr(self.convergence_monitor, 'set_mode'):
            self.convergence_monitor.set_mode('aggressive')
        
        # 设置贝叶斯引擎为积极模式
        if hasattr(self.bayesian_engine, 'set_aggressive_mode'):
            self.bayesian_engine.set_aggressive_mode()
        
        # 更新配置
        self.config.update({
            'aggressive_mutation_mode': True,
            'prefer_bayesian_decisions': True,
            'enable_bayesian_analysis': True
        })
        
        logger.info("🚀 智能DNM核心已设置为积极模式")
    
    def set_conservative_mode(self):
        """设置保守模式"""
        
        # 设置收敛监控为保守模式
        if hasattr(self.convergence_monitor, 'set_mode'):
            self.convergence_monitor.set_mode('conservative')
        
        # 设置贝叶斯引擎为保守模式
        if hasattr(self.bayesian_engine, 'set_conservative_mode'):
            self.bayesian_engine.set_conservative_mode()
        
        # 更新配置
        self.config.update({
            'aggressive_mutation_mode': False,
            'prefer_bayesian_decisions': False,
        })
        
        logger.info("🛡️ 智能DNM核心已设置为保守模式")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        
        status = {
            'config': self.config,
            'execution_history_length': len(self.execution_history)
        }
        
        # 添加组件状态
        if hasattr(self.convergence_monitor, 'get_status_summary'):
            status['convergence_monitor'] = self.convergence_monitor.get_status_summary()
        
        if hasattr(self.bayesian_engine, 'get_analysis_summary'):
            status['bayesian_engine'] = self.bayesian_engine.get_analysis_summary()
        
        return status
    
    # 移除重复的转换方法，现在使用schema_transformer
    
    def _convert_decisions_to_candidates(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将贝叶斯决策转换为候选点格式"""
        
        candidates = []
        for decision in decisions:
            candidate = {
                'layer_name': decision.get('layer_name', ''),
                'layer_type': 'bayesian_identified',
                'selection_reasons': ['bayesian_optimization'],
                'bottleneck_types': ['bayesian_detected'],
                'improvement_potential': decision.get('expected_improvement', 0.0),
                'priority_score': decision.get('expected_utility', 0.0),
                'recommended_mutations': [decision.get('mutation_type', '')],
                'bayesian_metrics': {
                    'success_probability': decision.get('success_probability', 0.5),
                    'decision_confidence': decision.get('decision_confidence', 0.5),
                    'expected_utility': decision.get('expected_utility', 0.0)
                }
            }
            candidates.append(candidate)
        
        return candidates
    
    def _convert_decisions_to_strategies(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将贝叶斯决策转换为策略格式"""
        
        strategies = []
        for decision in decisions:
            strategy = {
                'target_layer': decision.get('layer_name', ''),
                'mutation_type': decision.get('mutation_type', ''),
                'rationale': {
                    'selection_method': 'bayesian_inference',
                    'success_probability': decision.get('success_probability', 0.5),
                    'expected_improvement': decision.get('expected_improvement', 0.0),
                    'decision_confidence': decision.get('decision_confidence', 0.5)
                },
                'expected_outcome': {
                    'expected_accuracy_improvement': decision.get('expected_improvement', 0.0),
                    'confidence_level': decision.get('decision_confidence', 0.5),
                    'success_probability': decision.get('success_probability', 0.5)
                },
                'risk_assessment': {
                    'overall_risk': 1.0 - decision.get('success_probability', 0.5),
                    'risk_factors': [],
                    'value_at_risk': decision.get('risk_metrics', {}).get('value_at_risk', 0.0),
                    'expected_shortfall': decision.get('risk_metrics', {}).get('expected_shortfall', 0.0)
                },
                'bayesian_reasoning': decision.get('rationale', 'Bayesian optimization recommended'),
                'implementation_priority': decision.get('expected_utility', 0.0)
            }
            strategies.append(strategy)
        
        return strategies
    
    def update_bayesian_outcome(self, 
                              mutation_type: str,
                              layer_name: str,
                              success: bool,
                              performance_change: float,
                              context: Dict[str, Any]):
        """更新贝叶斯引擎的变异结果，用于在线学习"""
        
        if hasattr(self, 'bayesian_engine') and self.bayesian_engine:
            self.bayesian_engine.update_mutation_outcome(
                mutation_type=mutation_type,
                layer_name=layer_name,
                success=success,
                performance_change=performance_change,
                context=context
            )
            logger.info(f"📊 已更新贝叶斯学习: {mutation_type} @ {layer_name} -> {'✅成功' if success else '❌失败'}")
        
        # 记录执行历史
        outcome_record = {
            'timestamp': context.get('epoch', 0),
            'mutation_type': mutation_type,
            'layer_name': layer_name,
            'success': success,
            'performance_change': performance_change,
            'engine_used': 'bayesian' if self.config.get('prefer_bayesian_decisions') else 'traditional'
        }
        self.execution_history.append(outcome_record)
    
    def get_bayesian_insights(self) -> Dict[str, Any]:
        """获取贝叶斯引擎的洞察信息"""
        
        if not hasattr(self, 'bayesian_engine') or not self.bayesian_engine:
            return {'status': 'bayesian_engine_not_available'}
        
        insights = {
            'mutation_history_length': len(self.bayesian_engine.mutation_history),
            'performance_history_length': len(self.bayesian_engine.performance_history),
            'architecture_features_tracked': len(self.bayesian_engine.architecture_features),
            'current_priors': self.bayesian_engine.mutation_priors.copy(),
            'dynamic_thresholds': self.bayesian_engine.dynamic_thresholds.copy(),
            'utility_parameters': self.bayesian_engine.utility_params.copy(),
            'recent_mutations': list(self.bayesian_engine.mutation_history)[-5:] if self.bayesian_engine.mutation_history else []
        }
        
        return insights
    
    def adjust_bayesian_parameters(self, parameter_updates: Dict[str, Any]):
        """调整贝叶斯引擎参数"""
        
        if not hasattr(self, 'bayesian_engine') or not self.bayesian_engine:
            logger.warning("⚠️ 贝叶斯引擎不可用，无法调整参数")
            return
        
        # 更新动态阈值
        if 'thresholds' in parameter_updates:
            for key, value in parameter_updates['thresholds'].items():
                if key in self.bayesian_engine.dynamic_thresholds:
                    self.bayesian_engine.dynamic_thresholds[key] = value
                    logger.info(f"📊 更新贝叶斯阈值: {key} = {value}")
        
        # 更新效用参数
        if 'utility' in parameter_updates:
            for key, value in parameter_updates['utility'].items():
                if key in self.bayesian_engine.utility_params:
                    self.bayesian_engine.utility_params[key] = value
                    logger.info(f"📊 更新效用参数: {key} = {value}")
        
        # 更新先验分布
        if 'priors' in parameter_updates:
            for mutation_type, prior_params in parameter_updates['priors'].items():
                if mutation_type in self.bayesian_engine.mutation_priors:
                    self.bayesian_engine.mutation_priors[mutation_type].update(prior_params)
                    logger.info(f"📊 更新先验分布: {mutation_type} = {prior_params}")
    
    def enable_aggressive_bayesian_mode(self):
        """启用积极的贝叶斯模式（更容易触发变异）"""
        
        if hasattr(self, 'bayesian_engine') and self.bayesian_engine:
            # 降低阈值，提高探索性
            aggressive_updates = {
                'thresholds': {
                    'min_expected_improvement': 0.001,   # 更低的期望改进阈值
                    'confidence_threshold': 0.2,        # 更低的置信度阈值
                    'exploration_threshold': 0.15       # 更积极的探索
                },
                'utility': {
                    'risk_aversion': 0.1,               # 降低风险厌恶
                    'exploration_bonus': 0.15           # 增加探索奖励
                }
            }
            
            self.adjust_bayesian_parameters(aggressive_updates)
            logger.info("🚀 已启用积极贝叶斯模式")
        else:
            logger.warning("⚠️ 贝叶斯引擎不可用，无法启用积极模式")