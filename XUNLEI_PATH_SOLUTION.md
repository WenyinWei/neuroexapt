# 迅雷下载路径问题解决方案

## 问题描述

用户反馈：使用 Neuro Exapt 的迅雷下载功能时，迅雷弹出的下载窗口显示的保存路径是平时使用的默认路径，而不是项目指定的路径。

## 根本原因

迅雷的 `thunder://` URL协议调用无法直接指定保存路径，这是迅雷的设计限制。当使用URL协议启动下载时，迅雷会使用其默认配置的下载路径。

## 解决方案

### 1. 多层级调用策略

我们实现了多种迅雷调用方式，按优先级排序：

1. **COM接口调用** (最可靠)
   - 使用 `win32com.client` 直接调用迅雷COM对象
   - 可以精确指定保存路径和文件名
   - 需要安装 `pywin32` 包

2. **命令行参数调用**
   - 尝试使用迅雷的命令行参数指定路径
   - 支持度取决于迅雷版本

3. **URL协议调用** (兜底方案)
   - 使用 `thunder://` 协议启动下载
   - 无法指定路径，但兼容性最好
   - 提供清晰的用户提示

4. **.thunder文件生成**
   - 生成迅雷任务文件作为最终备选
   - 用户可以双击文件启动下载

### 2. 配置助手工具

创建了 `XunleiConfigHelper` 类，提供：

- **自动检测**：检测当前迅雷配置
- **自动设置**：通过注册表或配置文件设置默认下载路径
- **交互式配置**：用户友好的配置界面
- **配置指南**：详细的手动配置说明

### 3. 用户友好的提示

在下载过程中提供清晰的提示：

```
✅ 迅雷下载已启动 (URL协议): cifar-10-python.tar.gz
⚠️ URL协议无法指定保存路径，请手动设置
💡 建议保存到: D:\Repo\neuroexapt\datasets\cifar-10-python.tar.gz
💡 提示: 可以运行以下命令配置迅雷默认下载路径:
   python -m neuroexapt.utils.xunlei_config_helper
```

## 使用方法

### 自动配置 (推荐)

```bash
# 运行配置助手
python -m neuroexapt.utils.xunlei_config_helper

# 选择 'y' 自动设置下载路径为 ./datasets
```

### 手动配置

1. 打开迅雷
2. 点击设置图标 ⚙️
3. 选择"下载设置"
4. 设置默认下载目录为: `D:\Repo\neuroexapt\datasets`
5. 点击确定保存

### 验证配置

```bash
# 运行测试脚本
python test_xunlei_path.py
```

## 技术实现

### 核心类

- `XunleiDownloader`: 迅雷下载器主类
- `XunleiConfigHelper`: 配置助手类
- `XunleiDatasetDownloader`: 数据集下载器

### 关键方法

```python
def download_with_xunlei(self, url: str, save_path: str, filename: Optional[str] = None) -> bool:
    """多层级迅雷调用策略"""
    
def _try_xunlei_com_download(self, url: str, save_path: str, filename: str) -> bool:
    """COM接口调用"""
    
def set_download_path(self, new_path: str) -> bool:
    """设置迅雷默认下载路径"""
```

### 配置文件支持

- 注册表配置: `HKEY_CURRENT_USER\SOFTWARE\Thunder Network\Thunder`
- 配置文件: `config.ini`
- 项目配置: `.xunlei_config.json`

## 用户体验改进

### 1. 清晰的错误提示

- 区分不同类型的错误
- 提供具体的解决建议
- 显示期望的保存路径

### 2. 多种配置方式

- 自动配置 (推荐)
- 手动配置 (GUI)
- 注册表配置 (高级用户)

### 3. 验证和测试

- 提供测试脚本验证配置
- 检查下载文件位置
- 监控下载进度

## 兼容性

### 支持的迅雷版本

- 迅雷9及以上版本
- 迅雷极速版
- 迅雷X

### 支持的操作系统

- Windows (主要支持)
- macOS (基础支持)
- Linux (基础支持)

### 依赖包

- `pywin32` (可选，用于COM接口)
- `requests` (用于网络请求)
- `pathlib` (用于路径处理)

## 未来改进

### 1. 更智能的路径检测

- 自动检测项目结构
- 智能推荐下载路径
- 支持多项目配置

### 2. 更强大的COM接口

- 支持更多迅雷功能
- 批量下载管理
- 下载进度监控

### 3. 云端配置同步

- 用户配置云端保存
- 多设备配置同步
- 团队配置共享

## 总结

通过实现多层级调用策略、配置助手工具和用户友好的提示，我们成功解决了迅雷下载路径问题。用户现在可以：

1. **自动配置**迅雷默认下载路径
2. **获得清晰提示**关于保存位置
3. **使用多种备选方案**确保下载成功
4. **验证配置**是否正确

这个解决方案既保持了迅雷下载的便利性，又解决了路径指定的问题，为中国用户提供了更好的使用体验。 