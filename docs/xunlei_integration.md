# 迅雷 (Xunlei/Thunder) Integration Guide for Chinese Users

## 🇨🇳 概述 (Overview)

迅雷 (Xunlei/Thunder) 是中国最流行的下载管理器之一，使用P2P技术可以显著加速数据集下载。本指南将帮助中国用户使用迅雷来加速Neuro Exapt框架的数据集下载。

## 🚀 快速开始 (Quick Start)

### 1. 安装迅雷 (Install 迅雷)

**Windows:**
- 访问 https://www.xunlei.com/ 下载最新版本
- 或从Microsoft Store安装

**macOS:**
- 从Mac App Store安装
- 或访问 https://www.xunlei.com/ 下载

**Linux:**
- 访问 https://www.xunlei.com/ 下载Linux版本

### 2. 自动检测 (Auto-Detection)

Neuro Exapt会自动检测系统中安装的迅雷：

```python
from neuroexapt.utils.xunlei_downloader import XunleiDownloader

# 自动检测迅雷
downloader = XunleiDownloader()
if downloader.is_available:
    print(f"✅ 迅雷已检测到: {downloader.xunlei_path}")
else:
    print("❌ 未检测到迅雷，请先安装")
```

### 3. 下载数据集 (Download Datasets)

```python
from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader

# 创建下载器
downloader = XunleiDatasetDownloader(data_dir="./data")

# 下载CIFAR-10数据集
success = downloader.download_dataset('cifar10')
if success:
    print("✅ 下载已启动，请查看迅雷")
```

## 📋 支持的数据集 (Supported Datasets)

| 数据集 | 大小 | 描述 | 下载URL |
|--------|------|------|---------|
| CIFAR-10 | 162MB | 图像分类数据集 | https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz |
| CIFAR-100 | 161MB | 图像分类数据集 | https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz |
| MNIST | 9.4MB | 手写数字数据集 | http://yann.lecun.com/exdb/mnist/ |

## 🔧 使用方法 (Usage Methods)

### 方法1: 自动下载 (Automatic Download)

```python
# 最简单的方法
downloader = XunleiDatasetDownloader()
downloader.download_dataset('cifar10')
```

### 方法2: 手动下载 (Manual Download)

1. 打开迅雷
2. 复制数据集URL
3. 粘贴到迅雷下载对话框
4. 设置保存路径为 `./data`
5. 开始下载

### 方法3: 任务文件 (Task File)

```python
# 创建迅雷任务文件
downloader = XunleiDownloader()
thunder_file = downloader.create_download_task_file(
    url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    save_path="./data",
    filename="cifar-10-python.tar.gz"
)
# 双击.thunder文件即可在迅雷中打开
```

## ⚡ 迅雷优化设置 (迅雷 Optimization Settings)

### 基本设置 (Basic Settings)

1. **启用迅雷加速** (Enable 迅雷 Acceleration)
   - 打开迅雷设置
   - 找到"下载设置"
   - 启用"迅雷加速"

2. **设置下载路径** (Set Download Path)
   - 默认路径: `./data`
   - 确保有足够的磁盘空间

3. **连接数设置** (Connection Settings)
   - 建议设置: 10-20个连接
   - 根据网络情况调整

### 高级设置 (Advanced Settings)

1. **P2P加速** (P2P Acceleration)
   - 启用P2P下载
   - 允许上传以加速其他用户

2. **带宽限制** (Bandwidth Limit)
   - 根据网络情况设置
   - 避免影响其他网络活动

3. **下载优先级** (Download Priority)
   - 设置数据集下载为高优先级
   - 确保快速完成

## 🎯 性能对比 (Performance Comparison)

| 下载方式 | 速度 | 稳定性 | 推荐度 |
|----------|------|--------|--------|
| 直接下载 | 慢 (10-50KB/s) | 不稳定 | ⭐⭐ |
| 迅雷下载 | 快 (1-10MB/s) | 稳定 | ⭐⭐⭐⭐⭐ |
| 迅雷VIP | 极快 (10-50MB/s) | 非常稳定 | ⭐⭐⭐⭐⭐ |

## 🔍 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

**1. 迅雷未检测到**
```python
# 手动指定迅雷路径
downloader = XunleiDownloader(xunlei_path="C:/Program Files/Thunder Network/Thunder/Program/Thunder.exe")
```

**2. 下载速度慢**
- 检查网络连接
- 启用迅雷加速
- 考虑升级迅雷VIP

**3. 下载中断**
- 迅雷支持断点续传
- 重新启动下载即可继续

**4. 文件损坏**
```python
# 验证文件完整性
from neuroexapt.utils.xunlei_downloader import XunleiDatasetDownloader
downloader = XunleiDatasetDownloader()
status = downloader.get_status()
print(status['datasets']['cifar10'])
```

### 错误代码 (Error Codes)

| 错误代码 | 含义 | 解决方案 |
|----------|------|----------|
| 404 | 文件不存在 | 检查URL是否正确 |
| 403 | 访问被拒绝 | 尝试使用VPN |
| 500 | 服务器错误 | 稍后重试 |
| 连接超时 | 网络问题 | 检查网络连接 |

## 💡 最佳实践 (Best Practices)

### 1. 网络优化 (Network Optimization)

- 使用有线网络连接
- 关闭其他下载任务
- 选择网络空闲时段下载

### 2. 迅雷设置 (迅雷 Settings)

- 定期更新迅雷版本
- 启用所有加速功能
- 设置合理的下载限制

### 3. 文件管理 (File Management)

- 定期清理下载缓存
- 备份重要数据集
- 使用SSD存储以提高速度

## 🚀 高级功能 (Advanced Features)

### 批量下载 (Batch Download)

```python
# 批量下载多个数据集
datasets = ['cifar10', 'cifar100', 'mnist']
downloader = XunleiDatasetDownloader()

for dataset in datasets:
    print(f"开始下载 {dataset}...")
    downloader.download_dataset(dataset)
```

### 进度监控 (Progress Monitoring)

```python
# 监控下载进度
import time

downloader = XunleiDatasetDownloader()
downloader.download_dataset('cifar10', wait_for_completion=False)

while True:
    status = downloader.get_status()
    cifar10_status = status['datasets']['cifar10']
    
    if cifar10_status['downloaded'] and cifar10_status['complete']:
        print("✅ 下载完成!")
        break
    elif cifar10_status['downloaded']:
        print(f"⏳ 下载进度: {cifar10_status['progress']:.1f}%")
    
    time.sleep(5)
```

### 自定义配置 (Custom Configuration)

```python
# 自定义下载配置
class CustomXunleiDownloader(XunleiDatasetDownloader):
    def __init__(self):
        super().__init__(
            data_dir="./custom_data",
            xunlei_path="C:/Custom/Path/To/Thunder.exe"
        )
    
    def download_dataset(self, dataset_name, **kwargs):
        # 自定义下载逻辑
        print(f"使用自定义配置下载 {dataset_name}")
        return super().download_dataset(dataset_name, **kwargs)
```

## 📞 技术支持 (Technical Support)

### 获取帮助 (Getting Help)

1. **查看日志** (Check Logs)
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **报告问题** (Report Issues)
- GitHub Issues: https://github.com/yourusername/neuroexapt/issues
- 邮件支持: team@neuroexapt.ai

3. **社区讨论** (Community Discussion)
- GitHub Discussions: https://github.com/yourusername/neuroexapt/discussions

### 贡献代码 (Contributing)

欢迎贡献代码来改进迅雷集成功能：

1. Fork项目
2. 创建功能分支
3. 提交Pull Request
4. 等待代码审查

## 📚 相关资源 (Related Resources)

- [迅雷官方网站](https://www.xunlei.com/)
- [Neuro Exapt文档](docs/html/index.html)
- [PyTorch数据集](https://pytorch.org/vision/stable/datasets.html)
- [中国镜像站点](https://mirrors.tuna.tsinghua.edu.cn/)

---

*本指南专为中国用户设计，帮助您充分利用迅雷的加速功能来快速下载机器学习数据集。* 