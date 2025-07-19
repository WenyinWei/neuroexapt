#!/bin/bash

# NeuroExapt Documentation Build Script
# 用于本地构建和测试文档

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数定义
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "🧠 NeuroExapt Documentation Builder"
    echo "====================================="
    echo ""
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    log_success "Python 3 found: $(python3 --version)"
    
    # 检查Doxygen
    if ! command -v doxygen &> /dev/null; then
        log_warning "Doxygen not found. Installing..."
        
        # 尝试安装Doxygen
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y doxygen graphviz
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install doxygen graphviz
            else
                log_error "Please install Doxygen and Graphviz manually"
                exit 1
            fi
        else
            log_error "Please install Doxygen and Graphviz manually"
            exit 1
        fi
    fi
    log_success "Doxygen found: $(doxygen --version)"
    
    # 检查Python依赖
    log_info "Checking Python dependencies..."
    python3 -c "import pathlib, logging, subprocess" 2>/dev/null || {
        log_error "Required Python modules not available"
        exit 1
    }
    log_success "Python dependencies OK"
}

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # 安装基础依赖
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt || log_warning "Some requirements failed to install"
    fi
    
    # 安装文档生成依赖
    pip3 install markdown beautifulsoup4 lxml || {
        log_warning "Documentation dependencies installation failed"
    }
    
    log_success "Python dependencies installed"
}

clean_docs() {
    log_info "Cleaning old documentation..."
    
    if [ -d "docs/generated" ]; then
        rm -rf docs/generated
        log_success "Removed old generated docs"
    fi
    
    if [ -d "docs/temp" ]; then
        rm -rf docs/temp
        log_success "Removed temp docs"
    fi
}

generate_docs() {
    log_info "Generating documentation..."
    
    # 确保脚本可执行
    chmod +x docs/generate_docs.py
    
    # 运行文档生成
    if python3 docs/generate_docs.py --verbose; then
        log_success "Documentation generated successfully"
    else
        log_error "Documentation generation failed"
        exit 1
    fi
}

validate_docs() {
    log_info "Validating generated documentation..."
    
    # 检查关键文件
    required_files=(
        "docs/generated/html/index.html"
        "docs/generated/html/modules.html"
        "docs/generated/html/files.html"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "✓ $file exists"
        else
            log_error "✗ $file missing"
            return 1
        fi
    done
    
    # 计算文档大小
    if [ -d "docs/generated/html" ]; then
        size=$(du -sh docs/generated/html | cut -f1)
        log_info "Documentation size: $size"
    fi
    
    log_success "Documentation validation passed"
}

open_docs() {
    local index_file="docs/generated/html/index.html"
    
    if [ -f "$index_file" ]; then
        local full_path=$(realpath "$index_file")
        log_success "Documentation built successfully!"
        echo ""
        echo "📖 Open documentation:"
        echo "   file://$full_path"
        echo ""
        
        # 尝试自动打开浏览器
        if command -v xdg-open &> /dev/null; then
            log_info "Opening documentation in browser..."
            xdg-open "file://$full_path" &
        elif command -v open &> /dev/null; then
            log_info "Opening documentation in browser..."
            open "file://$full_path"
        elif command -v start &> /dev/null; then
            log_info "Opening documentation in browser..."
            start "file://$full_path"
        else
            log_info "Please open the above URL in your browser"
        fi
    else
        log_error "Documentation index file not found"
        exit 1
    fi
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -c, --clean       Clean only (don't build)"
    echo "  -q, --quick       Quick build (skip dependency checks)"
    echo "  -v, --validate    Validate only (don't build)"
    echo "  -o, --open        Open documentation after building"
    echo "  --no-deps         Skip dependency installation"
    echo ""
    echo "Examples:"
    echo "  $0                # Full build with all checks"
    echo "  $0 --quick        # Quick build for development"
    echo "  $0 --clean        # Clean generated files only"
    echo "  $0 --open         # Build and open in browser"
    echo ""
}

# 解析命令行参数
CLEAN_ONLY=false
QUICK_BUILD=false
VALIDATE_ONLY=false
OPEN_DOCS=false
SKIP_DEPS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN_ONLY=true
            shift
            ;;
        -q|--quick)
            QUICK_BUILD=true
            shift
            ;;
        -v|--validate)
            VALIDATE_ONLY=true
            shift
            ;;
        -o|--open)
            OPEN_DOCS=true
            shift
            ;;
        --no-deps)
            SKIP_DEPS=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主执行流程
main() {
    print_header
    
    # 如果只是清理
    if [ "$CLEAN_ONLY" = true ]; then
        clean_docs
        log_success "Cleanup completed"
        exit 0
    fi
    
    # 如果只是验证
    if [ "$VALIDATE_ONLY" = true ]; then
        if validate_docs; then
            log_success "Validation passed"
            exit 0
        else
            log_error "Validation failed"
            exit 1
        fi
    fi
    
    # 检查依赖（除非是快速构建）
    if [ "$QUICK_BUILD" = false ]; then
        check_dependencies
        
        if [ "$SKIP_DEPS" = false ]; then
            install_python_deps
        fi
    fi
    
    # 清理旧文档
    clean_docs
    
    # 生成文档
    generate_docs
    
    # 验证文档
    validate_docs
    
    # 打开文档（如果请求）
    if [ "$OPEN_DOCS" = true ]; then
        open_docs
    else
        log_success "Documentation build completed!"
        echo ""
        echo "📁 Generated files: docs/generated/html/"
        echo "🌐 Open file://$(realpath docs/generated/html/index.html) to view"
    fi
}

# 捕获中断信号
trap 'log_error "Build interrupted"; exit 1' INT TERM

# 运行主函数
main "$@"