# NeuroExapt WSL2 Ubuntu迁移脚本
# 用于将项目从Windows迁移到WSL2 Ubuntu环境以获得完整的Triton支持

Write-Host "🐧 NeuroExapt WSL2 Ubuntu迁移助手" -ForegroundColor Green
Write-Host "=" * 50

# 检查WSL状态
Write-Host "`n🔍 检查WSL状态..." -ForegroundColor Yellow
$wslStatus = wsl --status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ WSL未安装或未启用" -ForegroundColor Red
    Write-Host "💡 请先启用WSL功能:" -ForegroundColor Cyan
    Write-Host "   1. 以管理员身份运行PowerShell"
    Write-Host "   2. 执行: dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart"
    Write-Host "   3. 执行: dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart"
    Write-Host "   4. 重启电脑"
    Write-Host "   5. 下载并安装WSL2内核更新: https://aka.ms/wsl2kernel"
    exit 1
}

# 检查Ubuntu是否已安装
Write-Host "✅ WSL已启用" -ForegroundColor Green
$ubuntuInstalled = wsl -l -v | Select-String "Ubuntu"
if (-not $ubuntuInstalled) {
    Write-Host "`n📦 安装Ubuntu..." -ForegroundColor Yellow
    Write-Host "执行: wsl --install Ubuntu"
    wsl --install Ubuntu
    Write-Host "⚠️ Ubuntu安装完成后，请重新运行此脚本" -ForegroundColor Cyan
    Read-Host "按回车键退出"
    exit 0
} else {
    Write-Host "✅ Ubuntu已安装" -ForegroundColor Green
}

# 获取当前项目路径
$currentPath = Get-Location
Write-Host "`n📁 当前项目路径: $currentPath" -ForegroundColor Cyan

# 创建迁移脚本
$migrationScript = @"
#!/bin/bash
echo "🚀 NeuroExapt WSL2环境配置脚本"
echo "================================"

# 更新系统
echo "📦 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
echo "🔧 安装基础依赖..."
sudo apt install -y python3 python3-pip python3-venv git curl wget build-essential

# 安装NVIDIA驱动支持 (WSL2)
echo "🎮 配置NVIDIA CUDA支持..."
# 检查是否已有CUDA
if ! command -v nvcc &> /dev/null; then
    echo "安装CUDA工具链..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-2
    
    # 添加CUDA到PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
else
    echo "✅ CUDA已安装"
fi

# 创建Python虚拟环境
echo "🐍 创建Python虚拟环境..."
cd ~
python3 -m venv neuroexapt_env
source neuroexapt_env/bin/activate

# 安装PyTorch和Triton
echo "📦 安装PyTorch和Triton..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton

# 复制项目文件
echo "📂 复制项目文件..."
if [ -d "~/neuroexapt" ]; then
    echo "⚠️ 目录已存在，备份为neuroexapt_backup_$(date +%Y%m%d_%H%M%S)"
    mv ~/neuroexapt ~/neuroexapt_backup_$(date +%Y%m%d_%H%M%S)
fi

cp -r /mnt/d/Repo/neuroexapt ~/neuroexapt
cd ~/neuroexapt

# 安装项目依赖
echo "📦 安装项目依赖..."
pip install -r requirements.txt
pip install -e .

# 验证安装
echo "🧪 验证安装..."
python -c "
import torch
import triton
print('✅ PyTorch版本:', torch.__version__)
print('✅ CUDA可用:', torch.cuda.is_available())
print('✅ Triton版本:', triton.__version__)

# 测试Triton基本功能
import triton.language as tl
print('✅ Triton language可用')

# 简单的Triton内核测试
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

print('✅ Triton内核编译成功')
"

echo ""
echo "🎉 WSL2环境配置完成！"
echo "================================"
echo "📍 项目位置: ~/neuroexapt"
echo "🐍 激活环境: source ~/neuroexapt_env/bin/activate"
echo "🚀 运行测试: cd ~/neuroexapt && python check_triton.py"
echo ""
echo "💡 重要提示:"
echo "   1. 每次使用前需激活虚拟环境"
echo "   2. WSL2中GPU性能与原生Linux相近"
echo "   3. 文件在/mnt/d/可访问Windows文件"
echo ""
"@

# 保存迁移脚本到临时文件
$scriptPath = "$env:TEMP\neuroexapt_wsl_setup.sh"
$migrationScript | Out-File -FilePath $scriptPath -Encoding UTF8

Write-Host "`n🚀 开始WSL2环境配置..." -ForegroundColor Green
Write-Host "执行迁移脚本..." -ForegroundColor Yellow

# 复制脚本到WSL并执行
wsl -d Ubuntu -u root cp /mnt/c/Windows/Temp/neuroexapt_wsl_setup.sh /tmp/setup.sh
wsl -d Ubuntu -u root chmod +x /tmp/setup.sh
wsl -d Ubuntu -u root /tmp/setup.sh

Write-Host "`n✅ 迁移完成！" -ForegroundColor Green
Write-Host "🎯 下一步操作:" -ForegroundColor Cyan
Write-Host "   1. wsl -d Ubuntu  # 进入WSL2"
Write-Host "   2. source ~/neuroexapt_env/bin/activate  # 激活环境"
Write-Host "   3. cd ~/neuroexapt  # 进入项目目录"
Write-Host "   4. python check_triton.py  # 验证Triton"
Write-Host "   5. python examples/basic_classification.py --mode exapt  # 运行优化训练"

Write-Host "`n💡 性能对比建议:" -ForegroundColor Yellow
Write-Host "   Windows (仅CUDA): python examples/basic_classification.py --mode fixed"
Write-Host "   WSL2 (Triton+CUDA): python examples/basic_classification.py --mode exapt"

Read-Host "`n按回车键退出" 