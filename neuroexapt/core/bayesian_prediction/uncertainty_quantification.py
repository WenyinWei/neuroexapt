"""
不确定性量化模块

提供预测不确定性量化、置信区间估计等功能
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class UncertaintyQuantifier:
    """
    不确定性量化器
    
    提供预测不确定性的量化和分析功能，包括：
    - 认知不确定性（模型不确定性）
    - 偶然不确定性（数据噪声）
    - 置信区间估计
    - 预测可靠性评估
    """
    
    def __init__(self, monte_carlo_samples: int = 100):
        """
        初始化不确定性量化器
        
        Args:
            monte_carlo_samples: 蒙特卡罗采样数量
        """
        self.mc_samples = monte_carlo_samples
        self.confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        
    def quantify_prediction_uncertainty(self, 
                                      predictions: np.ndarray,
                                      model_variance: Optional[np.ndarray] = None,
                                      data_noise: Optional[float] = None) -> Dict[str, Any]:
        """
        量化预测不确定性
        
        Args:
            predictions: 预测结果数组
            model_variance: 模型方差（认知不确定性）
            data_noise: 数据噪声水平（偶然不确定性）
            
        Returns:
            不确定性量化结果
        """
        # 计算预测统计量
        mean_prediction = np.mean(predictions)
        prediction_std = np.std(predictions)
        
        # 认知不确定性（模型不确定性）
        if model_variance is not None:
            epistemic_uncertainty = np.mean(model_variance)
        else:
            epistemic_uncertainty = prediction_std**2
            
        # 偶然不确定性（数据噪声）
        if data_noise is not None:
            aleatoric_uncertainty = data_noise**2
        else:
            aleatoric_uncertainty = prediction_std**2 * 0.1  # 估计
            
        # 总不确定性
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # 置信区间
        confidence_intervals = self._calculate_confidence_intervals(
            predictions, mean_prediction, np.sqrt(total_uncertainty)
        )
        
        # 预测可靠性
        reliability_score = self._calculate_reliability_score(
            predictions, total_uncertainty
        )
        
        return {
            'mean_prediction': float(mean_prediction),
            'prediction_std': float(prediction_std),
            'epistemic_uncertainty': float(epistemic_uncertainty),
            'aleatoric_uncertainty': float(aleatoric_uncertainty),
            'total_uncertainty': float(total_uncertainty),
            'confidence_intervals': confidence_intervals,
            'reliability_score': float(reliability_score),
            'prediction_confidence': float(1.0 / (1.0 + total_uncertainty))
        }
    
    def _calculate_confidence_intervals(self, 
                                      predictions: np.ndarray,
                                      mean: float,
                                      std: float) -> Dict[str, Tuple[float, float]]:
        """计算置信区间"""
        intervals = {}
        
        for confidence_level in self.confidence_levels:
            # 使用正态分布近似
            z_score = self._get_z_score(confidence_level)
            lower = mean - z_score * std
            upper = mean + z_score * std
            
            # 也使用经验分位数
            empirical_lower = np.percentile(predictions, (1 - confidence_level) * 50)
            empirical_upper = np.percentile(predictions, (1 + confidence_level) * 50)
            
            # 取更保守的估计
            final_lower = min(lower, empirical_lower)
            final_upper = max(upper, empirical_upper)
            
            intervals[f'{confidence_level:.0%}'] = (float(final_lower), float(final_upper))
            
        return intervals
    
    def _get_z_score(self, confidence_level: float) -> float:
        """获取置信水平对应的Z分数"""
        z_scores = {
            0.68: 1.0,   # 1σ
            0.95: 1.96,  # 2σ
            0.99: 2.58   # 3σ
        }
        return z_scores.get(confidence_level, 1.96)
    
    def _calculate_reliability_score(self, 
                                   predictions: np.ndarray,
                                   total_uncertainty: float) -> float:
        """计算预测可靠性分数"""
        # 基于预测的一致性和不确定性
        prediction_consistency = 1.0 - (np.std(predictions) / (np.abs(np.mean(predictions)) + 1e-8))
        uncertainty_penalty = 1.0 / (1.0 + total_uncertainty)
        
        reliability = 0.5 * prediction_consistency + 0.5 * uncertainty_penalty
        return np.clip(reliability, 0.0, 1.0)
    
    def estimate_model_uncertainty(self, 
                                 model: torch.nn.Module,
                                 inputs: torch.Tensor,
                                 num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        估计模型不确定性（使用Dropout等方法）
        
        Args:
            model: PyTorch模型
            inputs: 输入数据
            num_samples: MC采样数量
            
        Returns:
            模型不确定性估计结果
        """
        if num_samples is None:
            num_samples = self.mc_samples
            
        model.train()  # 启用dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = model(inputs)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        variance_pred = np.var(predictions, axis=0)
        
        return {
            'mean_prediction': mean_pred,
            'variance_prediction': variance_pred,
            'std_prediction': np.sqrt(variance_pred),
            'predictions': predictions
        }
    
    def calibration_analysis(self, 
                           predicted_confidences: np.ndarray,
                           actual_accuracies: np.ndarray,
                           n_bins: int = 10) -> Dict[str, Any]:
        """
        校准分析 - 评估预测置信度的准确性
        
        Args:
            predicted_confidences: 预测的置信度
            actual_accuracies: 实际准确率
            n_bins: 分箱数量
            
        Returns:
            校准分析结果
        """
        # 将置信度分箱
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0.0
        reliability_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在当前箱中的样本
            in_bin = (predicted_confidences > bin_lower) & (predicted_confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # 该箱的平均置信度和准确率
                avg_confidence = predicted_confidences[in_bin].mean()
                avg_accuracy = actual_accuracies[in_bin].mean()
                
                # 校准误差
                calibration_error += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
                
                reliability_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'avg_confidence': float(avg_confidence),
                    'avg_accuracy': float(avg_accuracy),
                    'count': int(in_bin.sum())
                })
        
        # ECE (Expected Calibration Error)
        ece = calibration_error
        
        # MCE (Maximum Calibration Error)
        if reliability_data:
            mce = max([abs(d['avg_confidence'] - d['avg_accuracy']) for d in reliability_data])
        else:
            mce = 0.0
        
        return {
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'reliability_diagram_data': reliability_data,
            'is_well_calibrated': ece < 0.1  # 阈值可调
        }


class EnsembleUncertaintyQuantifier(UncertaintyQuantifier):
    """
    集成模型的不确定性量化器
    """
    
    def __init__(self, monte_carlo_samples: int = 100):
        super().__init__(monte_carlo_samples)
    
    def quantify_ensemble_uncertainty(self, 
                                    ensemble_predictions: List[np.ndarray]) -> Dict[str, Any]:
        """
        量化集成模型的不确定性
        
        Args:
            ensemble_predictions: 来自不同模型的预测列表
            
        Returns:
            集成不确定性量化结果
        """
        if not ensemble_predictions:
            raise ValueError("集成预测列表不能为空")
        
        # 将预测堆叠
        stacked_predictions = np.stack(ensemble_predictions, axis=0)
        
        # 计算集成统计量
        ensemble_mean = np.mean(stacked_predictions, axis=0)
        ensemble_std = np.std(stacked_predictions, axis=0)
        
        # 模型间差异（认知不确定性）
        inter_model_variance = np.var(stacked_predictions, axis=0)
        
        # 模型内不确定性（偶然不确定性的近似）
        intra_model_variance = np.mean([np.var(pred) for pred in ensemble_predictions])
        
        # 总不确定性
        total_uncertainty = inter_model_variance + intra_model_variance
        
        # 一致性分数
        consistency_score = 1.0 / (1.0 + ensemble_std.mean())
        
        return {
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'inter_model_variance': inter_model_variance,
            'intra_model_variance': float(intra_model_variance),
            'total_uncertainty': total_uncertainty,
            'consistency_score': float(consistency_score),
            'num_models': len(ensemble_predictions)
        }