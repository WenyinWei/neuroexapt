# NeuroExapt Documentation System

这个目录包含NeuroExapt项目的完整文档生成系统。

## 📁 目录结构

```
docs/
├── README.md                 # 本文件 - 文档系统说明
├── Doxyfile                  # Doxygen配置文件
├── DoxygenLayout.xml         # Doxygen布局配置
├── header.html               # 自定义HTML头部
├── footer.html               # 自定义HTML尾部
├── custom.css                # 自定义样式表
├── generate_docs.py          # 主文档生成脚本
├── build.sh                  # 本地构建脚本
├── generated/                # 生成的文档（自动创建）
│   └── html/                 # HTML文档输出
├── temp/                     # 临时文件（自动创建）
├── images/                   # 文档图片资源
└── mainpage.md               # 主页内容
```

## 🚀 快速开始

### 本地构建文档

```bash
# 完整构建（推荐首次使用）
./docs/build.sh

# 快速构建（开发时使用）
./docs/build.sh --quick

# 构建并自动打开浏览器
./docs/build.sh --open

# 仅清理生成的文件
./docs/build.sh --clean
```

### 手动生成

```bash
# 使用Python脚本
python docs/generate_docs.py --verbose

# 直接使用Doxygen
doxygen docs/Doxyfile
```

## 🔧 系统特性

### 1. 自动文档整合
- **Markdown文件整合**: 自动收集并分类项目中的所有markdown文件
- **智能分类**: 按照内容类型自动分组（架构、性能、修复、指南等）
- **示例代码执行**: 自动运行examples中的代码并生成结果
- **API文档生成**: 从Python代码注释自动生成API文档

### 2. 现代化界面
- **响应式设计**: 支持桌面和移动设备
- **深色模式**: 自动检测用户偏好
- **美观主题**: 渐变色彩和现代排版
- **交互功能**: 代码复制、返回顶部、搜索增强

### 3. GitHub Actions集成
- **自动构建**: 每次推送到主分支时自动生成文档
- **GitHub Pages部署**: 自动部署到GitHub Pages
- **定时更新**: 每日自动更新文档
- **构建状态通知**: 详细的构建报告和状态

## 📖 文档类型

### 自动分类的文档
1. **🔍 项目概览** - README、项目介绍等
2. **🏗️ 架构与框架** - DNM框架、网络架构等
3. **⚡ 性能与基准** - 性能优化、准确率分析等
4. **📋 指南与教程** - 安装指南、使用教程等
5. **🚀 优化与CUDA** - GPU优化、CUDA实现等
6. **🔧 修复与解决方案** - Bug修复、问题解决等
7. **👨‍💻 开发与改进** - 开发日志、代码改进等

### 生成的文档页面
- **API参考** - 完整的类和函数文档
- **模块文档** - 按模块组织的文档
- **示例代码** - 可运行的代码示例
- **测试结果** - 自动化测试输出
- **文件浏览** - 源代码浏览器

## ⚙️ 配置选项

### Doxygen配置
主要配置在`Doxyfile`中：
- **项目信息**: 名称、版本、描述
- **输入源**: Python代码、Markdown文件
- **输出格式**: HTML（已启用），PDF（已禁用）
- **主题定制**: 自定义CSS和布局

### Python脚本配置
`generate_docs.py`支持以下选项：
- `--project-root`: 项目根目录路径
- `--verbose`: 详细输出模式

### 构建脚本选项
`build.sh`支持多种构建模式：
- `--quick`: 快速构建（跳过依赖检查）
- `--clean`: 仅清理文件
- `--open`: 构建后自动打开浏览器
- `--no-deps`: 跳过依赖安装

## 🔨 开发指南

### 添加新的文档类型
1. 在`generate_docs.py`中的`category_rules`字典添加新类别
2. 更新`category_titles`字典添加显示名称
3. 重新构建文档

### 自定义样式
1. 编辑`custom.css`文件
2. 修改`header.html`和`footer.html`模板
3. 更新`DoxygenLayout.xml`布局配置

### 添加新的示例
1. 在`examples/`目录添加Python文件
2. 确保代码可以独立运行
3. 文档生成时会自动包含并执行

## 🚀 GitHub Actions

### 工作流触发条件
- **推送**: 推送到main/master分支
- **拉取请求**: PR到main/master分支
- **定时任务**: 每日凌晨2点自动更新
- **手动触发**: 在Actions页面手动运行

### 部署流程
1. **代码检出**: 获取最新代码
2. **环境设置**: 安装Python和Doxygen
3. **依赖安装**: 安装项目依赖
4. **文档生成**: 运行文档生成脚本
5. **验证检查**: 验证生成的文档
6. **部署上传**: 部署到GitHub Pages

### 访问文档
部署完成后，文档将在以下地址可用：
```
https://[username].github.io/[repository-name]/
```

## 🔍 故障排除

### 常见问题

**Doxygen未找到**
```bash
# Ubuntu/Debian
sudo apt-get install doxygen graphviz

# macOS
brew install doxygen graphviz

# Windows
# 下载并安装Doxygen from https://www.doxygen.nl/download.html
```

**Python依赖问题**
```bash
# 安装文档生成依赖
pip install markdown beautifulsoup4 lxml

# 如果有requirements.txt
pip install -r requirements.txt
```

**权限问题**
```bash
# 确保脚本可执行
chmod +x docs/build.sh
chmod +x docs/generate_docs.py
```

### 调试模式
使用详细输出查看详细信息：
```bash
python docs/generate_docs.py --verbose
```

## 📊 统计信息

文档系统会自动生成以下统计：
- 处理的Markdown文件数量
- 生成的HTML页面数量
- 文档总大小
- 构建时间
- 代码示例执行结果

## 🤝 贡献指南

1. **添加文档**: 在相应目录添加Markdown文件
2. **改进模板**: 修改HTML模板和CSS样式
3. **扩展功能**: 在`generate_docs.py`中添加新功能
4. **测试构建**: 使用`./docs/build.sh`测试本地构建
5. **提交变更**: 提交时会自动触发文档更新

## 📞 支持

如有问题，请：
1. 查看构建日志和错误信息
2. 检查依赖是否正确安装
3. 在GitHub Issues中报告问题
4. 参考本文档的故障排除部分

---

**构建状态**: 查看[GitHub Actions](../../actions)页面获取最新构建状态

**在线文档**: 访问[GitHub Pages](https://[username].github.io/[repository-name]/)查看最新文档