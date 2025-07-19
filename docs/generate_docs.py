#!/usr/bin/env python3
"""
NeuroExapt Documentation Generator

这个脚本负责：
1. 整合根目录下的所有markdown文档
2. 运行examples和tests中的代码示例
3. 生成Doxygen文档
4. 优化文档结构和内容
"""

import os
import sys
import subprocess
import shutil
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentationGenerator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.docs_dir = self.project_root / "docs"
        self.generated_dir = self.docs_dir / "generated"
        self.temp_dir = self.docs_dir / "temp"
        
        # 创建必要的目录
        self.docs_dir.mkdir(exist_ok=True)
        self.generated_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
    def clean_old_docs(self):
        """清理旧的生成文档"""
        logger.info("🧹 Cleaning old documentation...")
        
        if self.generated_dir.exists():
            shutil.rmtree(self.generated_dir)
        self.generated_dir.mkdir(exist_ok=True)
        
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
            
    def collect_markdown_files(self) -> List[Path]:
        """收集项目中所有的markdown文件"""
        logger.info("📝 Collecting markdown files...")
        
        md_files = []
        
        # 根目录的markdown文件
        root_md_files = list(self.project_root.glob("*.md"))
        md_files.extend(root_md_files)
        
        # docs目录中的markdown文件
        docs_md_files = list(self.docs_dir.glob("*.md"))
        md_files.extend(docs_md_files)
        
        # examples目录中的markdown文件
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            examples_md_files = list(examples_dir.glob("*.md"))
            md_files.extend(examples_md_files)
            
        logger.info(f"Found {len(md_files)} markdown files")
        return md_files
    
    def categorize_markdown_files(self, md_files: List[Path]) -> Dict[str, List[Path]]:
        """将markdown文件按类别分组"""
        logger.info("📂 Categorizing markdown files...")
        
        categories = {
            "overview": [],
            "architecture": [],
            "performance": [],
            "fixes": [],
            "guides": [],
            "development": [],
            "optimization": [],
            "other": []
        }
        
        # 定义分类规则
        category_rules = {
            "overview": ["README", "mainpage", "project"],
            "architecture": ["ARCHITECTURE", "DNM", "FRAMEWORK", "MORPHOGENESIS", "EVOLUTION"],
            "performance": ["PERFORMANCE", "OPTIMIZATION", "BENCHMARK", "ACCURACY"],
            "fixes": ["FIX", "BUG", "ERROR", "ISSUE", "SOLUTION"],
            "guides": ["GUIDE", "TUTORIAL", "SETUP", "INSTALLATION"],
            "development": ["DEVELOPMENT", "SOURCERY", "IMPROVEMENTS"],
            "optimization": ["OPTIMIZATION", "CUDA", "GPU", "TRITON"]
        }
        
        for md_file in md_files:
            categorized = False
            filename_upper = md_file.name.upper()
            
            for category, keywords in category_rules.items():
                if any(keyword in filename_upper for keyword in keywords):
                    categories[category].append(md_file)
                    categorized = True
                    break
            
            if not categorized:
                categories["other"].append(md_file)
        
        # 打印分类结果
        for category, files in categories.items():
            if files:
                logger.info(f"  {category}: {len(files)} files")
                
        return categories
    
    def create_master_documentation(self, categories: Dict[str, List[Path]]):
        """创建主文档页面，整合所有markdown内容"""
        logger.info("📚 Creating master documentation...")
        
        # 创建主页面
        main_doc = self.temp_dir / "documentation_overview.md"
        
        with open(main_doc, 'w', encoding='utf-8') as f:
            f.write("""# NeuroExapt Complete Documentation

\\page documentation_overview Documentation Overview

Welcome to the comprehensive documentation for **NeuroExapt** - an advanced Neural Architecture Search and Dynamic Morphogenesis framework.

## 🚀 Quick Start

- [Main README](README.md) - Project overview and quick start guide
- [Examples](examples.html) - Code examples and usage patterns
- [API Reference](modules.html) - Complete API documentation

## 📖 Documentation Sections

""")
            
            # 为每个类别创建章节
            section_order = ["overview", "architecture", "performance", "guides", "optimization", "fixes", "development", "other"]
            
            for category in section_order:
                files = categories.get(category, [])
                if not files:
                    continue
                    
                # 创建类别标题
                category_titles = {
                    "overview": "🔍 Project Overview",
                    "architecture": "🏗️ Architecture & Framework",
                    "performance": "⚡ Performance & Benchmarks", 
                    "guides": "📋 Guides & Tutorials",
                    "optimization": "🚀 Optimization & CUDA",
                    "fixes": "🔧 Fixes & Solutions",
                    "development": "👨‍💻 Development & Improvements",
                    "other": "📄 Additional Documentation"
                }
                
                f.write(f"### {category_titles.get(category, category.title())}\n\n")
                
                # 列出该类别下的文档
                for md_file in sorted(files):
                    # 读取文件的第一行作为描述
                    try:
                        with open(md_file, 'r', encoding='utf-8') as md:
                            first_line = md.readline().strip()
                            if first_line.startswith('#'):
                                title = first_line.lstrip('#').strip()
                            else:
                                title = md_file.stem.replace('_', ' ').title()
                    except:
                        title = md_file.stem.replace('_', ' ').title()
                    
                    # 创建相对路径
                    relative_path = os.path.relpath(md_file, self.project_root)
                    f.write(f"- [{title}]({relative_path})\n")
                
                f.write("\n")
            
            # 添加API文档链接
            f.write("""
## 🔧 Technical Reference

### Core Modules
- [Core Framework](group__core.html) - Main framework components
- [Neural Morphogenesis](group__morphogenesis.html) - Dynamic neural morphogenesis
- [Architecture Search](group__architecture.html) - Neural architecture search
- [Optimization](group__optimization.html) - Performance optimization
- [CUDA Operations](group__cuda.html) - CUDA acceleration

### Examples & Tests
- [Example Scripts](examples.html) - Runnable code examples
- [Test Results](test_results.html) - Automated test outcomes
- [Benchmarks](benchmarks.html) - Performance benchmarks

## 📊 Project Statistics

Generated on: """ + str(__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + """

For the latest updates, visit our [GitHub Repository](https://github.com/your-username/neuroexapt).
""")
    
    def run_example_tests(self) -> Dict[str, str]:
        """运行examples中的代码并收集输出"""
        logger.info("🧪 Running example tests...")
        
        examples_dir = self.project_root / "examples"
        results = {}
        
        if not examples_dir.exists():
            logger.warning("Examples directory not found")
            return results
        
        # 找到所有Python示例文件
        example_files = list(examples_dir.glob("*.py"))
        
        # 排除一些不适合自动运行的文件
        exclude_patterns = [
            "advanced_dnm_demo.py",  # 需要大量计算资源
            "*gpu*",  # 需要GPU
            "*cuda*",  # 需要CUDA
            "*download*",  # 需要网络
        ]
        
        safe_examples = []
        for ex_file in example_files:
            excluded = False
            for pattern in exclude_patterns:
                if ex_file.match(pattern):
                    excluded = True
                    break
            if not excluded:
                safe_examples.append(ex_file)
        
        logger.info(f"Found {len(safe_examples)} safe example files to run")
        
        for example_file in safe_examples[:5]:  # 限制运行数量
            logger.info(f"Running {example_file.name}...")
            
            try:
                # 设置超时和环境
                env = os.environ.copy()
                env['PYTHONPATH'] = str(self.project_root)
                
                result = subprocess.run(
                    [sys.executable, str(example_file)],
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60秒超时
                    cwd=str(self.project_root),
                    env=env
                )
                
                if result.returncode == 0:
                    results[example_file.name] = f"✅ Success\n\nOutput:\n```\n{result.stdout[:1000]}...\n```"
                else:
                    results[example_file.name] = f"❌ Failed (return code: {result.returncode})\n\nError:\n```\n{result.stderr[:500]}...\n```"
                    
            except subprocess.TimeoutExpired:
                results[example_file.name] = "⏱️ Timeout (60s limit exceeded)"
            except Exception as e:
                results[example_file.name] = f"💥 Exception: {str(e)}"
        
        return results
    
    def create_test_results_page(self, test_results: Dict[str, str]):
        """创建测试结果页面"""
        logger.info("📊 Creating test results page...")
        
        results_page = self.temp_dir / "test_results.md"
        
        with open(results_page, 'w', encoding='utf-8') as f:
            f.write("""# Test Results

\\page test_results Test Results

This page shows the results of running example scripts automatically during documentation generation.

""")
            
            for example_name, result in test_results.items():
                f.write(f"## {example_name}\n\n{result}\n\n---\n\n")
            
            if not test_results:
                f.write("No test results available. Tests may have been skipped due to missing dependencies.\n")
    
    def enhance_python_docstrings(self):
        """增强Python文件的文档字符串以便Doxygen处理"""
        logger.info("📝 Enhancing Python docstrings...")
        
        python_files = list(self.project_root.glob("neuroexapt/**/*.py"))
        
        for py_file in python_files:
            if "test" in str(py_file) or "legacy" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 添加模块级别的文档组
                if "\\defgroup" not in content and "class " in content:
                    module_name = py_file.stem
                    group_name = f"group_{module_name}"
                    
                    # 在文件开头添加组定义
                    group_header = f'''"""
\\defgroup {group_name} {module_name.replace('_', ' ').title()}
\\ingroup core
{py_file.stem.replace('_', ' ').title()} module for NeuroExapt framework.
"""

'''
                    
                    # 查找第一个非注释行
                    lines = content.split('\n')
                    insert_line = 0
                    for i, line in enumerate(lines):
                        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                            insert_line = i
                            break
                    
                    lines.insert(insert_line, group_header)
                    enhanced_content = '\n'.join(lines)
                    
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(enhanced_content)
                        
            except Exception as e:
                logger.warning(f"Failed to enhance {py_file}: {e}")
    
    def run_doxygen(self):
        """运行Doxygen生成文档"""
        logger.info("🔧 Running Doxygen...")
        
        doxyfile = self.docs_dir / "Doxyfile"
        
        if not doxyfile.exists():
            logger.error("Doxygen configuration file not found!")
            return False
        
        try:
            result = subprocess.run(
                ["doxygen", str(doxyfile)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Doxygen completed successfully")
                return True
            else:
                logger.error(f"❌ Doxygen failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("❌ Doxygen not found. Please install doxygen.")
            return False
        except Exception as e:
            logger.error(f"❌ Error running Doxygen: {e}")
            return False
    
    def create_images_directory(self):
        """创建图片目录并复制相关图片"""
        logger.info("🖼️ Setting up images directory...")
        
        images_dir = self.docs_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 创建一个简单的项目logo (SVG)
        logo_svg = images_dir / "logo.svg"
        with open(logo_svg, 'w') as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
  </defs>
  <circle cx="60" cy="60" r="50" fill="url(#grad1)"/>
  <text x="60" y="70" font-family="Arial, sans-serif" font-size="48" fill="white" text-anchor="middle">🧠</text>
</svg>''')
    
    def post_process_html(self):
        """后处理生成的HTML文件"""
        logger.info("🎨 Post-processing HTML files...")
        
        html_dir = self.generated_dir / "html"
        if not html_dir.exists():
            logger.warning("Generated HTML directory not found")
            return
        
        # 查找所有HTML文件
        html_files = list(html_dir.glob("*.html"))
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 添加额外的meta标签
                if '<head>' in content:
                    meta_tags = '''
    <meta name="description" content="NeuroExapt - Advanced Neural Architecture Search and Dynamic Morphogenesis Framework">
    <meta name="keywords" content="neural architecture search, dynamic morphogenesis, deep learning, AI, PyTorch">
    <meta name="author" content="NeuroExapt Team">
    <meta property="og:title" content="NeuroExapt Documentation">
    <meta property="og:description" content="Advanced Neural Architecture Search Framework">
    <meta property="og:type" content="website">
'''
                    content = content.replace('<head>', '<head>' + meta_tags)
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            except Exception as e:
                logger.warning(f"Failed to post-process {html_file}: {e}")
    
    def generate_documentation(self):
        """主文档生成流程"""
        logger.info("🚀 Starting documentation generation...")
        
        try:
            # 1. 清理旧文档
            self.clean_old_docs()
            
            # 2. 收集和分类markdown文件
            md_files = self.collect_markdown_files()
            categories = self.categorize_markdown_files(md_files)
            
            # 3. 创建主文档
            self.create_master_documentation(categories)
            
            # 4. 运行示例测试
            test_results = self.run_example_tests()
            self.create_test_results_page(test_results)
            
            # 5. 设置图片目录
            self.create_images_directory()
            
            # 6. 增强Python文档字符串
            self.enhance_python_docstrings()
            
            # 7. 运行Doxygen
            success = self.run_doxygen()
            
            if success:
                # 8. 后处理HTML
                self.post_process_html()
                
                logger.info("✅ Documentation generation completed successfully!")
                logger.info(f"📁 Generated documentation is in: {self.generated_dir / 'html'}")
                
                # 显示主页路径
                index_html = self.generated_dir / "html" / "index.html"
                if index_html.exists():
                    logger.info(f"🌐 Open documentation: file://{index_html}")
                    
                return True
            else:
                logger.error("❌ Documentation generation failed")
                return False
                
        except Exception as e:
            logger.error(f"💥 Error during documentation generation: {e}")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NeuroExapt documentation')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    generator = DocumentationGenerator(args.project_root)
    success = generator.generate_documentation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()