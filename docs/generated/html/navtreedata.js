/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "NeuroExapt", "index.html", [
    [ "DNM Framework Examples", "index.html", "index" ],
    [ "Advanced Dataset Loader", "md_neuroexapt_2utils_2README__dataset__loader.html", [
      [ "🌟 Key Features", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md1", [
        [ "🚀 P2P Acceleration", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md2", null ],
        [ "🏪 Intelligent Caching", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md3", null ],
        [ "🌍 Multiple Mirror Support", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md4", null ],
        [ "🔄 Robust Error Handling", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md5", null ]
      ] ],
      [ "📦 Installation", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md6", null ],
      [ "🚀 Quick Start", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md7", [
        [ "Basic Usage", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md8", null ],
        [ "Advanced Configuration", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md9", null ]
      ] ],
      [ "📊 Supported Datasets", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md10", [
        [ "CIFAR-10", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md11", null ],
        [ "CIFAR-100", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md12", null ],
        [ "MNIST", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md13", null ]
      ] ],
      [ "🔧 API Reference", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md14", [
        [ "AdvancedDatasetLoader", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md15", [
          [ "Constructor", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md16", null ],
          [ "Methods", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md17", [
            [ "<tt>get_cifar10_dataloaders()</tt>", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md18", null ],
            [ "<tt>get_cifar100_dataloaders()</tt>", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md19", null ],
            [ "<tt>download_dataset()</tt>", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md20", null ],
            [ "<tt>clear_cache()</tt>", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md21", null ],
            [ "<tt>get_cache_info()</tt>", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md22", null ]
          ] ]
        ] ]
      ] ],
      [ "🌍 Mirror Configuration", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md23", [
        [ "Chinese Mirrors (Priority 1-2)", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md24", null ],
        [ "International Mirrors (Priority 3-4)", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md25", null ]
      ] ],
      [ "📈 Performance Optimization", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md26", [
        [ "For Chinese Users", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md27", null ],
        [ "For International Users", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md28", null ]
      ] ],
      [ "🔍 Monitoring and Debugging", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md29", [
        [ "Cache Information", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md30", null ],
        [ "Mirror Status", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md31", null ],
        [ "Logging", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md32", null ]
      ] ],
      [ "🛠️ Troubleshooting", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md33", [
        [ "Common Issues", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md34", [
          [ "Slow Downloads", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md35", null ],
          [ "Cache Corruption", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md36", null ],
          [ "Network Timeouts", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md37", null ],
          [ "P2P Failures", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md38", null ]
        ] ],
        [ "Testing", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md39", null ]
      ] ],
      [ "🔄 Migration from Standard Loaders", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md40", [
        [ "Before (Standard PyTorch)", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md41", null ],
        [ "After (Advanced Loader)", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md42", null ]
      ] ],
      [ "📊 Performance Benchmarks", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md43", [
        [ "Download Speeds (China)", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md44", null ],
        [ "Cache Benefits", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md45", null ]
      ] ],
      [ "🤝 Contributing", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md46", null ],
      [ "📄 License", "md_neuroexapt_2utils_2README__dataset__loader.html#autotoc_md47", null ]
    ] ],
    [ "🧠 NeuroExapt 文档系统部署完成! 🎉", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html", [
      [ "🚀 系统概览", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md69", null ],
      [ "📁 创建的文件结构", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md70", null ],
      [ "🎯 核心功能特性", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md71", [
        [ "1. 📚 智能文档整合", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md72", null ],
        [ "2. 🎨 现代化用户界面", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md73", null ],
        [ "3. 🤖 自动化工作流", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md74", null ],
        [ "4. 🧪 代码示例执行", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md75", null ]
      ] ],
      [ "🚀 立即开始使用", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md76", [
        [ "本地测试", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md77", null ],
        [ "GitHub Actions部署", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md78", null ]
      ] ],
      [ "📊 整合的文档内容", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md79", [
        [ "🔍 项目概览 (3+ 文档)", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md80", null ],
        [ "🏗️ 架构文档 (10+ 文档)", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md81", null ],
        [ "⚡ 性能文档 (8+ 文档)", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md82", null ],
        [ "🔧 修复文档 (15+ 文档)", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md83", null ],
        [ "🚀 优化文档 (5+ 文档)", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md84", null ]
      ] ],
      [ "🎨 界面预览特性", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md85", [
        [ "头部设计", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md86", null ],
        [ "导航菜单", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md87", null ],
        [ "尾部信息", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md88", null ]
      ] ],
      [ "🔧 自定义和扩展", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md89", [
        [ "添加新文档类型", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md90", null ],
        [ "自定义样式", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md91", null ],
        [ "扩展功能", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md92", null ]
      ] ],
      [ "📈 预期效果", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md93", [
        [ "文档网站功能", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md94", null ],
        [ "自动化收益", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md95", null ]
      ] ],
      [ "🎯 下一步操作", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md96", [
        [ "立即可做", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md97", null ],
        [ "优化建议", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md98", null ]
      ] ],
      [ "🔗 相关链接", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md99", null ],
      [ "🎉 完成状态", "md_docs_2changelogs_2DOCUMENTATION__SYSTEM__SETUP.html#autotoc_md100", null ]
    ] ],
    [ "GitHub Actions版本更新记录", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html", [
      [ "🎯 更新目的", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md102", null ],
      [ "❌ 问题描述", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md103", null ],
      [ "🔧 已更新的Actions", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md104", [
        [ "1. actions/upload-artifact", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md105", null ],
        [ "2. actions/upload-pages-artifact", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md106", null ],
        [ "3. actions/deploy-pages", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md107", null ],
        [ "4. actions/setup-python", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md108", null ]
      ] ],
      [ "📊 版本兼容性矩阵", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md109", null ],
      [ "🚨 重要的破坏性变更", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md110", [
        [ "upload-artifact v4的重要变更：", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md111", null ],
        [ "迁移示例：", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md112", null ]
      ] ],
      [ "🔍 验证检查", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md113", [
        [ "更新后的工作流验证：", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md114", null ],
        [ "性能改进期望：", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md115", null ]
      ] ],
      [ "📝 配置文件变更", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md116", [
        [ "<tt>.github/workflows/docs.yml</tt>更新内容：", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md117", null ]
      ] ],
      [ "🔮 未来维护建议", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md118", [
        [ "1. 定期检查Actions版本", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md119", null ],
        [ "2. 版本固定策略", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md120", null ],
        [ "3. 监控和告警", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md121", null ]
      ] ],
      [ "📚 参考资料", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md122", null ],
      [ "✅ 更新状态", "md_docs_2changelogs_2GITHUB__ACTIONS__UPDATE.html#autotoc_md123", null ]
    ] ],
    [ "NeuroExapt Logger Method Inconsistencies - FIXED! 🎉", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html", [
      [ "🐛 Original Problem", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md126", null ],
      [ "🔧 Root Cause Analysis", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md127", [
        [ "Missing Methods", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md128", null ],
        [ "Inconsistent Usage", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md129", null ]
      ] ],
      [ "✅ Comprehensive Fix Implementation", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md130", [
        [ "1. Enhanced ConfigurableLogger Class", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md131", null ],
        [ "2. Enhanced DebugPrinter Class (Backward Compatibility)", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md132", null ],
        [ "3. Fixed Inconsistent Debug Calls", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md133", null ],
        [ "4. Removed Duplicate Code", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md134", null ]
      ] ],
      [ "🧪 Testing & Verification", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md135", [
        [ "Methods Available", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md136", null ],
        [ "Syntax Verification", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md137", null ]
      ] ],
      [ "🚀 Impact & Benefits", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md138", [
        [ "1. Reliability", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md139", null ],
        [ "2. Maintainability", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md140", null ],
        [ "3. Functionality", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md141", null ],
        [ "4. Backward Compatibility", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md142", null ]
      ] ],
      [ "🎯 Error Resolution", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md143", [
        [ "Original Error", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md144", null ],
        [ "Solution Applied", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md145", null ],
        [ "Expected Result", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md146", null ]
      ] ],
      [ "🔄 Deployment Instructions", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md147", [
        [ "For Users", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md148", null ],
        [ "For Developers", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md149", null ]
      ] ],
      [ "📋 Files Modified", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md150", null ],
      [ "✨ Next Steps", "md_docs_2changelogs_2LOGGER__METHOD__FIXES__SUMMARY.html#autotoc_md151", null ]
    ] ],
    [ "NeuroExapt Project Structure Refactor Summary", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html", [
      [ "🎯 Goals Achieved", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md154", [
        [ "✅ 1. Circular Import Resolution", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md155", null ],
        [ "✅ 2. Project Structure Cleanup", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md156", null ]
      ] ],
      [ "🔧 Technical Changes", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md157", [
        [ "New Logging Infrastructure", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md158", null ],
        [ "Import Graph Restructure", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md159", null ]
      ] ],
      [ "📁 New Directory Structure", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md160", [
        [ "Core Package", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md161", null ],
        [ "Examples (Essential Demos)", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md162", null ],
        [ "Test Organization", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md163", null ],
        [ "Utilities", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md164", null ]
      ] ],
      [ "🚀 Key Benefits", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md165", [
        [ "1. Import Reliability", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md166", null ],
        [ "2. Development Experience", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md167", null ],
        [ "3. Maintainability", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md168", null ]
      ] ],
      [ "🧪 Verification", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md169", null ],
      [ "📈 Performance Impact", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md170", [
        [ "Positive Changes:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md171", null ],
        [ "Backward Compatibility:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md172", null ]
      ] ],
      [ "🔄 Migration Guide", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md173", [
        [ "For Developers:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md174", null ],
        [ "For Users:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md175", null ]
      ] ],
      [ "🎛️ Configuration Options", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md176", [
        [ "Environment Variables:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md177", null ]
      ] ],
      [ "📋 File Movement Summary", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md178", [
        [ "Moved to <tt>test/legacy/</tt>:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md179", null ],
        [ "Moved to <tt>utilities/</tt>:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md180", null ],
        [ "Moved to <tt>examples/</tt>:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md181", null ],
        [ "Kept in Root:", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md182", null ]
      ] ],
      [ "✨ Next Steps", "md_docs_2changelogs_2PROJECT__STRUCTURE__REFACTOR.html#autotoc_md183", null ]
    ] ],
    [ "Sourcery Code Quality Optimizations Applied", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html", [
      [ "📋 Overview", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md186", null ],
      [ "🔧 Optimizations Applied", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md187", [
        [ "1. Environment Variable Standardization", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md188", null ],
        [ "2. Logger Instance Caching", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md189", null ],
        [ "3. Documentation Organization", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md190", null ],
        [ "4. Documentation System Updates", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md191", null ]
      ] ],
      [ "📊 Impact Summary", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md192", [
        [ "Code Quality Improvements", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md193", null ],
        [ "Documentation Improvements", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md194", null ],
        [ "Developer Experience", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md195", null ]
      ] ],
      [ "🎯 Best Practices Established", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md196", [
        [ "1. Documentation Hierarchy", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md197", null ],
        [ "2. Environment Variables", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md198", null ],
        [ "3. Logger Management", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md199", null ],
        [ "4. Code Organization", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md200", null ]
      ] ],
      [ "🔄 Future Recommendations", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md201", [
        [ "Immediate", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md202", null ],
        [ "Long-term", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md203", null ]
      ] ],
      [ "📈 Metrics", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md204", [
        [ "Before Optimization", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md205", null ],
        [ "After Optimization", "md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md206", null ]
      ] ]
    ] ],
    [ "Symbol Glossary", "symbols.html", [
      [ "Core Information Theory Symbols", "symbols.html#autotoc_md269", [
        [ "Basic Information Measures", "symbols.html#autotoc_md270", null ],
        [ "Network-Specific Variables", "symbols.html#autotoc_md271", null ]
      ] ],
      [ "Structural Evolution Symbols", "symbols.html#autotoc_md272", [
        [ "Layer Importance and Redundancy", "symbols.html#autotoc_md273", null ],
        [ "Task-Specific Parameters", "symbols.html#autotoc_md274", null ],
        [ "Evolution Dynamics", "symbols.html#autotoc_md275", null ]
      ] ],
      [ "Entropy Control Symbols", "symbols.html#autotoc_md276", [
        [ "Adaptive Thresholding", "symbols.html#autotoc_md277", null ],
        [ "Complexity Estimation", "symbols.html#autotoc_md278", null ]
      ] ],
      [ "Discrete Parameter Optimization", "symbols.html#autotoc_md279", [
        [ "Continuous Relaxation", "symbols.html#autotoc_md280", null ],
        [ "Parameter Ranges", "symbols.html#autotoc_md281", null ]
      ] ],
      [ "Mathematical Operations", "symbols.html#autotoc_md282", [
        [ "Probability and Statistics", "symbols.html#autotoc_md283", null ],
        [ "Optimization and Calculus", "symbols.html#autotoc_md284", null ]
      ] ],
      [ "Convergence Theory Symbols", "symbols.html#autotoc_md285", [
        [ "Theoretical Bounds", "symbols.html#autotoc_md286", null ],
        [ "Complexity Classes", "symbols.html#autotoc_md287", null ]
      ] ],
      [ "Implementation Constants", "symbols.html#autotoc_md288", [
        [ "Numerical Stability", "symbols.html#autotoc_md289", null ],
        [ "Algorithmic Parameters", "symbols.html#autotoc_md290", null ]
      ] ],
      [ "Special Notation", "symbols.html#autotoc_md291", [
        [ "Set Theory and Logic", "symbols.html#autotoc_md292", null ],
        [ "Functional Notation", "symbols.html#autotoc_md293", null ]
      ] ],
      [ "Index and Summation Conventions", "symbols.html#autotoc_md294", [
        [ "Standard Indices", "symbols.html#autotoc_md295", null ],
        [ "Summation Notation", "symbols.html#autotoc_md296", null ]
      ] ],
      [ "Units and Scales", "symbols.html#autotoc_md297", [
        [ "Information Units", "symbols.html#autotoc_md298", null ],
        [ "Time Scales", "symbols.html#autotoc_md299", null ]
      ] ]
    ] ],
    [ "Documentation Overview", "documentation_overview.html", [
      [ "🚀 Quick Start", "documentation_overview.html#autotoc_md302", null ],
      [ "📖 Documentation Sections", "documentation_overview.html#autotoc_md303", [
        [ "🔍 Project Overview", "documentation_overview.html#autotoc_md304", null ],
        [ "🏗️ Architecture & Framework", "documentation_overview.html#autotoc_md305", null ],
        [ "⚡ Performance & Benchmarks", "documentation_overview.html#autotoc_md306", null ],
        [ "📋 Guides & Tutorials", "documentation_overview.html#autotoc_md307", null ],
        [ "🔧 Fixes & Solutions", "documentation_overview.html#autotoc_md308", null ],
        [ "👨‍💻 Development & Improvements", "documentation_overview.html#autotoc_md309", null ],
        [ "📄 Additional Documentation", "documentation_overview.html#autotoc_md310", null ]
      ] ],
      [ "🔧 Technical Reference", "documentation_overview.html#autotoc_md311", [
        [ "Core Modules", "documentation_overview.html#autotoc_md312", null ],
        [ "Examples & Tests", "documentation_overview.html#autotoc_md313", null ]
      ] ],
      [ "📊 Project Statistics", "documentation_overview.html#autotoc_md314", null ]
    ] ],
    [ "Test Results", "test_results.html", [
      [ "demo_neuroexapt_v3.py", "test_results.html#autotoc_md316", null ],
      [ "basic_classification.py", "test_results.html#autotoc_md318", null ],
      [ "simple_batchnorm_debug.py", "test_results.html#autotoc_md320", null ],
      [ "dynamic_architecture_evolution.py", "test_results.html#autotoc_md322", null ],
      [ "separated_training_example.py", "test_results.html#autotoc_md324", null ]
    ] ],
    [ "Theoretical Foundation", "theory.html", [
      [ "1. Information-Theoretic Foundations", "theory.html#autotoc_md326", [
        [ "1.1 Core Principles", "theory.html#autotoc_md327", null ],
        [ "1.2 Layer Importance Evaluation", "theory.html#autotoc_md328", null ],
        [ "1.3 Network Redundancy Calculation", "theory.html#autotoc_md329", null ]
      ] ],
      [ "2. Discrete Parameter Optimization", "theory.html#autotoc_md330", [
        [ "2.1 Continuous Relaxation", "theory.html#autotoc_md331", null ],
        [ "2.2 Parameter Initialization", "theory.html#autotoc_md332", null ]
      ] ],
      [ "3. Dynamic Evolution Mechanisms", "theory.html#autotoc_md333", [
        [ "3.1 Structural Entropy Balance", "theory.html#autotoc_md334", null ],
        [ "3.2 Adaptive Entropy Threshold", "theory.html#autotoc_md335", null ]
      ] ],
      [ "4. Convergence Theory", "theory.html#autotoc_md336", [
        [ "4.1 Main Convergence Theorem", "theory.html#autotoc_md337", null ],
        [ "4.2 Proof Sketch", "theory.html#autotoc_md338", null ],
        [ "4.3 Convergence Rate Analysis", "theory.html#autotoc_md339", null ]
      ] ],
      [ "5. Operational Algorithms", "theory.html#autotoc_md340", [
        [ "5.1 Information Assessment Algorithm", "theory.html#autotoc_md341", null ],
        [ "5.2 Structural Evolution Algorithm", "theory.html#autotoc_md342", null ],
        [ "5.3 Adaptive Threshold Update", "theory.html#autotoc_md343", null ]
      ] ],
      [ "6. Implementation Considerations", "theory.html#autotoc_md344", [
        [ "6.1 Numerical Stability", "theory.html#autotoc_md345", null ],
        [ "6.2 Computational Efficiency", "theory.html#autotoc_md346", null ],
        [ "6.3 Hyperparameter Sensitivity", "theory.html#autotoc_md347", null ]
      ] ],
      [ "7. Extensions and Future Directions", "theory.html#autotoc_md348", [
        [ "7.1 Multi-Task Learning", "theory.html#autotoc_md349", null ]
      ] ]
    ] ],
    [ "迅雷 (Xunlei/Thunder) Integration Guide for Chinese Users", "md_docs_2xunlei__integration.html", [
      [ "🇨🇳 概述 (Overview)", "md_docs_2xunlei__integration.html#autotoc_md354", null ],
      [ "🚀 快速开始 (Quick Start)", "md_docs_2xunlei__integration.html#autotoc_md355", [
        [ "1. 安装迅雷 (Install 迅雷)", "md_docs_2xunlei__integration.html#autotoc_md356", null ],
        [ "2. 自动检测 (Auto-Detection)", "md_docs_2xunlei__integration.html#autotoc_md357", null ],
        [ "3. 下载数据集 (Download Datasets)", "md_docs_2xunlei__integration.html#autotoc_md358", null ]
      ] ],
      [ "📋 支持的数据集 (Supported Datasets)", "md_docs_2xunlei__integration.html#autotoc_md359", null ],
      [ "🔧 使用方法 (Usage Methods)", "md_docs_2xunlei__integration.html#autotoc_md360", [
        [ "方法1: 自动下载 (Automatic Download)", "md_docs_2xunlei__integration.html#autotoc_md361", null ],
        [ "方法2: 手动下载 (Manual Download)", "md_docs_2xunlei__integration.html#autotoc_md362", null ],
        [ "方法3: 任务文件 (Task File)", "md_docs_2xunlei__integration.html#autotoc_md363", null ]
      ] ],
      [ "⚡ 迅雷优化设置 (迅雷 Optimization Settings)", "md_docs_2xunlei__integration.html#autotoc_md364", [
        [ "基本设置 (Basic Settings)", "md_docs_2xunlei__integration.html#autotoc_md365", null ],
        [ "高级设置 (Advanced Settings)", "md_docs_2xunlei__integration.html#autotoc_md366", null ]
      ] ],
      [ "🎯 性能对比 (Performance Comparison)", "md_docs_2xunlei__integration.html#autotoc_md367", null ],
      [ "🔍 故障排除 (Troubleshooting)", "md_docs_2xunlei__integration.html#autotoc_md368", [
        [ "常见问题 (Common Issues)", "md_docs_2xunlei__integration.html#autotoc_md369", null ],
        [ "错误代码 (Error Codes)", "md_docs_2xunlei__integration.html#autotoc_md370", null ]
      ] ],
      [ "💡 最佳实践 (Best Practices)", "md_docs_2xunlei__integration.html#autotoc_md371", [
        [ "1. 网络优化 (Network Optimization)", "md_docs_2xunlei__integration.html#autotoc_md372", null ],
        [ "2. 迅雷设置 (迅雷 Settings)", "md_docs_2xunlei__integration.html#autotoc_md373", null ],
        [ "3. 文件管理 (File Management)", "md_docs_2xunlei__integration.html#autotoc_md374", null ]
      ] ],
      [ "🚀 高级功能 (Advanced Features)", "md_docs_2xunlei__integration.html#autotoc_md375", [
        [ "批量下载 (Batch Download)", "md_docs_2xunlei__integration.html#autotoc_md376", null ],
        [ "进度监控 (Progress Monitoring)", "md_docs_2xunlei__integration.html#autotoc_md377", null ],
        [ "自定义配置 (Custom Configuration)", "md_docs_2xunlei__integration.html#autotoc_md378", null ]
      ] ],
      [ "📞 技术支持 (Technical Support)", "md_docs_2xunlei__integration.html#autotoc_md379", [
        [ "获取帮助 (Getting Help)", "md_docs_2xunlei__integration.html#autotoc_md380", null ],
        [ "贡献代码 (Contributing)", "md_docs_2xunlei__integration.html#autotoc_md381", null ]
      ] ],
      [ "📚 相关资源 (Related Resources)", "md_docs_2xunlei__integration.html#autotoc_md382", null ]
    ] ],
    [ "迅雷下载路径配置指南", "md_docs_2xunlei__setup__guide.html", [
      [ "问题描述", "md_docs_2xunlei__setup__guide.html#autotoc_md385", null ],
      [ "解决方案", "md_docs_2xunlei__setup__guide.html#autotoc_md386", [
        [ "方法1: 自动配置 (推荐)", "md_docs_2xunlei__setup__guide.html#autotoc_md387", null ],
        [ "方法2: 手动配置", "md_docs_2xunlei__setup__guide.html#autotoc_md388", null ],
        [ "方法3: 注册表配置", "md_docs_2xunlei__setup__guide.html#autotoc_md389", null ]
      ] ],
      [ "验证配置", "md_docs_2xunlei__setup__guide.html#autotoc_md390", null ],
      [ "故障排除", "md_docs_2xunlei__setup__guide.html#autotoc_md391", [
        [ "如果配置助手无法自动设置", "md_docs_2xunlei__setup__guide.html#autotoc_md392", null ],
        [ "如果迅雷仍然使用旧路径", "md_docs_2xunlei__setup__guide.html#autotoc_md393", null ]
      ] ],
      [ "高级配置", "md_docs_2xunlei__setup__guide.html#autotoc_md394", [
        [ "为不同项目设置不同路径", "md_docs_2xunlei__setup__guide.html#autotoc_md395", null ],
        [ "批量下载配置", "md_docs_2xunlei__setup__guide.html#autotoc_md396", null ]
      ] ],
      [ "技术说明", "md_docs_2xunlei__setup__guide.html#autotoc_md397", [
        [ "为什么URL协议无法指定路径", "md_docs_2xunlei__setup__guide.html#autotoc_md398", null ],
        [ "配置优先级", "md_docs_2xunlei__setup__guide.html#autotoc_md399", null ],
        [ "支持的迅雷版本", "md_docs_2xunlei__setup__guide.html#autotoc_md400", null ]
      ] ],
      [ "联系支持", "md_docs_2xunlei__setup__guide.html#autotoc_md401", null ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"advanced__dnm__demo_8py.html",
"classexamples_1_1basic__classification_1_1FixedNetwork.html",
"classexamples_1_1demo__intelligent__evolution_1_1AdaptiveCNN.html#ad8d59a479955c45bb1f30a80f991d723",
"classexamples_1_1dynamic__architecture__evolution_1_1EvolvableBlock.html#ac42e6069abd7a2c329773ea18ca5e4aa",
"classneuroexapt_1_1core_1_1advanced__morphogenesis_1_1AdvancedMorphogenesisExecutor.html#a3dbc71d138ffa9a015f391388761fa51",
"classneuroexapt_1_1core_1_1architecture__mutator_1_1ArchitectureMutator.html#ae09d81820f07969fd7ba02d51ea7da84",
"classneuroexapt_1_1core_1_1dnm__connection__growth_1_1DNMConnectionGrowth.html#aa980732e0dacda1212c7c7c94fcea4c4",
"classneuroexapt_1_1core_1_1dnm__layer__analyzer_1_1LayerPerformanceAnalyzer.html#a4c7aebf71f84ac6ae0e029b46d497896",
"classneuroexapt_1_1core_1_1enhanced__dnm__framework_1_1EnhancedDNMFramework.html#a6bc215f65e432109d0617812a8f3c47d",
"classneuroexapt_1_1core_1_1fast__operations_1_1FastMixedOp.html#aada1511ecefad7f502dde7f13d885356",
"classneuroexapt_1_1core_1_1model_1_1Network.html#a8dda474ad5c06ee0c305d1dcbe9fee0b",
"classneuroexapt_1_1core_1_1optimized__architect_1_1ArchitectureSpaceOptimizer.html#ad583dd0fd65f2b8e815bd0a97c566ef6",
"classneuroexapt_1_1core_1_1simple__architect_1_1SimpleArchitect.html#a6462c5593a9fcdb0dcc71b529a9b498a",
"classneuroexapt_1_1math_1_1pareto__optimization_1_1ModelFitnessEvaluator.html#ab57294bb056b5ddfabb253408f967dd2",
"classneuroexapt_1_1trainer__v3_1_1TrainerV3.html#ae708901396e7938fbaec098ad10bdd1c",
"classneuroexapt_1_1utils_1_1visualization_1_1AutoLayout.html#ab42738786d6ea3796316a7694ad2f155",
"dataset__loader_8py.html#a4ba5fa5715400e20acc3baacbaa9fa80",
"functions_func_f.html",
"md_docs_2changelogs_2SOURCERY__OPTIMIZATIONS.html#autotoc_md190",
"namespaceexamples_1_1demo__triton__cuda__optimization.html#a2b82a56529d6bdd06d6447b630e31af7",
"namespaceneuroexapt_1_1core_1_1logging__utils.html",
"robust__classification_8py.html#a402f523f3ca9b46029d711677a709adf"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';