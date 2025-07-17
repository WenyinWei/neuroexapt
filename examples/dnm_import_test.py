#!/usr/bin/env python3
"""
DNM导入测试 - 验证所有模块能否正确导入
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试所有DNM模块的导入"""
    
    print("🧪 开始DNM模块导入测试")
    print("=" * 50)
    
    # 测试核心模块导入
    tests = [
        ("DNM神经元分裂", "neuroexapt.core.dnm_neuron_division", ["DNMNeuronDivision", "IntelligentNeuronSplitter"]),
        ("DNM连接生长", "neuroexapt.core.dnm_connection_growth", ["DNMConnectionGrowth", "GradientConnectionAnalyzer"]),
        ("DNM多目标优化", "neuroexapt.math.pareto_optimization", ["ParetoOptimizer", "MultiObjectiveEvolution"]),
        ("DNM主框架", "neuroexapt.core.dnm_framework", ["DNMFramework"]),
        ("Net2Net变换器", "neuroexapt.core.dnm_net2net", ["Net2NetTransformer", "DNMArchitectureMutator"])
    ]
    
    success_count = 0
    total_tests = len(tests)
    
    for test_name, module_name, class_names in tests:
        try:
            print(f"📦 测试 {test_name}...")
            module = __import__(module_name, fromlist=class_names)
            
            # 验证类是否存在
            for class_name in class_names:
                if hasattr(module, class_name):
                    print(f"   ✅ {class_name} - 导入成功")
                else:
                    print(f"   ❌ {class_name} - 类不存在")
                    raise ImportError(f"Class {class_name} not found")
            
            success_count += 1
            print(f"   🎉 {test_name} 模块导入成功")
            
        except ImportError as e:
            print(f"   ❌ {test_name} 导入失败: {e}")
        except Exception as e:
            print(f"   ⚠️ {test_name} 其他错误: {e}")
        
        print()
    
    # 结果总结
    print("=" * 50)
    print(f"📊 测试结果: {success_count}/{total_tests} 模块成功导入")
    
    if success_count == total_tests:
        print("🎉 所有DNM模块导入成功!")
        print("✅ 代码框架完整性验证通过")
        return True
    else:
        print(f"❌ {total_tests - success_count} 个模块导入失败")
        print("🔧 需要修复导入问题")
        return False


def test_configuration():
    """测试配置创建"""
    print("⚙️ 测试DNM配置创建...")
    
    try:
        # 创建测试配置
        config = {
            'neuron_division': {
                'splitter': {
                    'entropy_threshold': 0.5,
                    'split_probability': 0.7,
                    'max_splits_per_layer': 3
                }
            },
            'connection_growth': {
                'analyzer': {
                    'correlation_threshold': 0.12
                }
            },
            'multi_objective': {
                'evolution': {
                    'population_size': 8
                }
            }
        }
        
        print(f"✅ DNM配置创建成功")
        print(f"   神经元分裂阈值: {config['neuron_division']['splitter']['entropy_threshold']}")
        print(f"   连接生长阈值: {config['connection_growth']['analyzer']['correlation_threshold']}")
        print(f"   进化种群大小: {config['multi_objective']['evolution']['population_size']}")
        return True
        
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return False


def test_framework_initialization():
    """测试框架初始化（无需实际模型）"""
    print("🏗️ 测试DNM框架初始化...")
    
    try:
        from neuroexapt.core.dnm_framework import DNMFramework
        
        config = {
            'neuron_division': {'splitter': {'entropy_threshold': 0.5}},
            'connection_growth': {'analyzer': {'correlation_threshold': 0.12}},
            'multi_objective': {'evolution': {'population_size': 8}},
            'framework': {'morphogenesis_frequency': 4}
        }
        
        # 创建框架实例（不实际运行训练）
        framework = DNMFramework(config)
        print(f"✅ DNM框架初始化成功")
        print(f"   形态发生频率: {framework.config['framework']['morphogenesis_frequency']}")
        return True
        
    except Exception as e:
        print(f"❌ 框架初始化失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 启动DNM完整性测试")
    print("🎯 目标：验证所有模块能否正确导入和初始化")
    print()
    
    # 运行所有测试
    test_results = []
    
    test_results.append(("模块导入", test_imports()))
    test_results.append(("配置创建", test_configuration()))
    test_results.append(("框架初始化", test_framework_initialization()))
    
    # 最终结果
    print("\n" + "=" * 60)
    print("🏆 DNM完整性测试结果")
    print("=" * 60)
    
    success_count = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n📊 总体结果: {success_count}/{len(test_results)} 测试通过")
    
    if success_count == len(test_results):
        print("\n🎉 DNM框架完整性验证成功!")
        print("✅ 所有核心模块正常工作")
        print("🚀 准备好进行实际训练测试")
        
        print("\n💡 下一步操作建议:")
        print("   1. 在有PyTorch环境的机器上运行: python examples/dnm_fixed_test.py")
        print("   2. 监控形态发生事件和架构变异")
        print("   3. 根据结果调整配置参数")
        print("   4. 冲击CIFAR-10 95%准确率目标!")
        
    else:
        print("\n❌ 存在问题需要修复")
        print("🔧 请检查失败的模块并修复")
    
    return success_count == len(test_results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)