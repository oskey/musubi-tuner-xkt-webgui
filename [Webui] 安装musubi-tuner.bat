@echo off
chcp 65001 > nul
echo =================================================
echo Musubi Tuner 纯净安装脚本（CUDA 12.4）
echo =================================================

:: -------------------------------
:: 1. 检查 Python 版本（官方要求 3.10+）
:: -------------------------------
set PYVER=
for /f %%v in ('python -c "import sys; print(sys.version[:5])" 2^>nul') do set PYVER=%%v
if "%PYVER%"=="" (
    echo [错误] 未检测到 Python，请先安装 Python 3.10 或更高版本。
    pause
    exit /b
)
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set PYMAJOR=%%a & set PYMINOR=%%b
if %PYMAJOR% LSS 3 if %PYMINOR% LSS 10 (
    echo [错误] Python 版本需 3.10+，当前版本为 %PYVER%。
    pause
    exit /b
)
echo ✅ 检测到 Python %PYVER%（符合官方要求）

:: -------------------------------
:: 2. 处理虚拟环境（默认不删除，询问用户）
:: -------------------------------
echo.
echo [1] 虚拟环境处理
set "DELETE_VENV=N"
set /p USERINPUT=是否删除现有虚拟环境并重建？(Y/N，默认N) 
if /i "%USERINPUT%"=="Y" set "DELETE_VENV=Y"

:: 根据用户选择处理虚拟环境
if "%DELETE_VENV%"=="Y" (
    echo 正在删除旧虚拟环境...
    if exist venv rmdir /s /q venv
    echo 正在创建新虚拟环境...
    python -m venv venv
    echo ✅ 虚拟环境已重建
) else (
    if not exist venv (
        echo 未检测到虚拟环境，正在创建...
        python -m venv venv
        echo ✅ 虚拟环境已创建
    ) else (
        echo ✅ 保留现有虚拟环境
        echo 检测到现有环境，跳转到安装结果验证...
        call venv\Scripts\activate.bat
        goto :VERIFY_INSTALLATION
    )
)

:: 激活虚拟环境
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [错误] 虚拟环境激活失败，请手动执行：venv\Scripts\activate.bat
    pause
    exit /b
)
echo ✅ 虚拟环境已激活

:: -------------------------------
:: 3. 清理虚拟环境内的旧依赖（仅保留旧环境时需要）
:: -------------------------------
echo [2] 清理环境依赖...
if "%DELETE_VENV%"=="N" (
    :: 只有保留旧环境时，才需要卸载里面的旧包
    echo 正在卸载虚拟环境内的旧版本依赖...
    pip uninstall -y torch torchvision torchaudio xformers
) else (
    :: 新环境无需卸载（本身就是空的）
    echo 新环境无需清理旧依赖
)
echo ✅ 依赖清理完成

:: -------------------------------
:: 4. 升级 pip
:: -------------------------------
echo [3] 升级 pip 到最新版本...
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
echo ✅ pip 升级完成

:: -------------------------------
:: 5. 安装 PyTorch（官方推荐）
:: -------------------------------
echo [4] 安装 PyTorch + torchvision（CUDA 12.4 版本）...
echo 指定版本：torch==2.5.1+cu124, torchvision==0.20.1+cu124
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% neq 0 (
    echo [错误] PyTorch 安装失败，建议手动执行：
    echo pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
    pause
    exit /b
)
echo ✅ PyTorch + torchvision 安装完成

:: -------------------------------
:: 6. 安装 Musubi Tuner 核心依赖
:: -------------------------------
echo [5] 安装 Musubi Tuner 本地依赖...
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo [错误] 核心依赖安装失败，请确保当前目录包含 setup.py
    pause
    exit /b
)
echo ✅ 核心依赖安装完成

echo 安装 TensorBoard（用于训练日志可视化）...
pip install tensorboard -i https://mirrors.aliyun.com/pypi/simple/
pip install flask  -i https://mirrors.aliyun.com/pypi/simple/
pip install flask-socketio  -i https://mirrors.aliyun.com/pypi/simple/

if %errorlevel% neq 0 (
    echo [警告] TensorBoard 安装失败，可能影响日志可视化功能
) else (
    echo ✅ TensorBoard 安装完成
)

:: -------------------------------
:: 7. 验证安装结果
:VERIFY_INSTALLATION
:: -------------------------------
echo.
echo =================================================
echo [6] 安装结果验证
echo =================================================
python -c "import torch; print(f'PyTorch 版本：{torch.__version__}（需 ≥2.5.1）')"
python -c "import torch; print(f'CUDA 支持：{torch.cuda.is_available()}（需为 True）')"
python -c "import torch; print(f'CUDA 版本：{torch.version.cuda if torch.cuda.is_available() else "未检测到"}')"
python -c "import torchvision; print(f'torchvision 版本：{torchvision.__version__}')"

echo.
echo =================================================
echo 是否安装 SageAttention？(Y/N，默认Y)
echo SageAttention 可以显著提升注意力机制的性能
echo =================================================
set /p install_sage="请选择 (Y/N): "
if "%install_sage%"=="" set install_sage=Y
if /i "%install_sage%"=="Y" (
    goto :INSTALL_SAGE
) else (
    echo 跳过 SageAttention 安装
    goto :FINAL_COMPLETE
)

:INSTALL_SAGE
echo.
echo =================================================
echo [7] 安装 SageAttention
echo =================================================
set SAGE_WHL=sageattention-2.1.1+cu124torch2.5.1-cp310-cp310-win_amd64.whl
set SAGE_URL=https://github.com/sdbds/SageAttention-for-windows/releases/download/2.1.1/%SAGE_WHL%
set SAGE_TMP=%SAGE_WHL%.tmp

if exist "%SAGE_WHL%" (
    echo ✅ 检测到本地文件：%SAGE_WHL%
    echo 跳过下载，直接安装...
) else (
    echo 正在下载 SageAttention...
    echo 下载地址：%SAGE_URL%
    powershell -Command "Invoke-WebRequest -Uri '%SAGE_URL%' -OutFile '%SAGE_TMP%'"
    if %errorlevel% neq 0 (
        echo [错误] SageAttention 下载失败
        pause
        exit /b
    )
    ren "%SAGE_TMP%" "%SAGE_WHL%"
    echo ✅ SageAttention 下载完成
)

echo 安装 triton 包...
set "triton_whl=triton-3.0.0-cp310-cp310-win_amd64.whl"
set "triton_url=https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp310-cp310-win_amd64.whl"
set "triton_tmp=%triton_whl%.tmp"

echo 检查本地是否已有Windows版triton文件...
if exist "%triton_whl%" (
    echo ✅ 检测到本地文件：%triton_whl%
    pip install "%triton_whl%" --force-reinstall
    if %errorlevel% equ 0 (
        echo ✅ triton 安装成功（本地文件）
        goto :install_sageattention
    ) else (
        echo ❌ 本地文件安装失败，尝试重新下载...
        del "%triton_whl%" 2>nul
    )
)

echo 正在从 HuggingFace 下载 triton...
echo 下载地址：%triton_url%
powershell -Command "Invoke-WebRequest -Uri '%triton_url%' -OutFile '%triton_tmp%'"
if %errorlevel% neq 0 (
    echo ❌ triton 下载失败！
    echo 请手动下载triton wheel文件到项目根目录后重新运行安装脚本
    echo 下载地址: https://huggingface.co/oskey/musubi-tuner-xkt-webgui/resolve/main/triton-3.0.0-cp310-cp310-win_amd64.whl
    pause
    exit /b 1
) else (
    ren "%triton_tmp%" "%triton_whl%"
    echo ✅ triton 下载完成
    pip install "%triton_whl%" --force-reinstall
    if %errorlevel% equ 0 (
        echo ✅ triton 安装成功（下载文件）
    ) else (
        echo ❌ triton 安装失败！
        echo 请检查下载的wheel文件是否完整或重新下载
        pause
        exit /b 1
    )
)

:install_sageattention
echo 正在安装 SageAttention...
pip install "%SAGE_WHL%" --force-reinstall
if %errorlevel% neq 0 (
    echo [错误] SageAttention 安装失败
    pause
    exit /b
)

echo 验证安装结果...
echo 检查 triton 模块...
python -c "import triton; print('Triton version: ' + triton.__version__)" 
if %errorlevel% neq 0 (
    echo [警告] triton 导入失败
) else (
    echo ✅ triton 验证通过
)

echo 检查 SageAttention 模块...
python -c "import sageattention; print('✅ SageAttention 导入成功')" 
if %errorlevel% neq 0 (
    echo [警告] SageAttention 导入失败，可能安装不完整
) else (
    echo ✅ SageAttention 验证通过
)

:FINAL_COMPLETE
echo.
echo =================================================
echo ✅ 所有安装步骤完成！
echo 提示：运行时请确保虚拟环境已激活（venv\Scripts\activate.bat）
echo =================================================
pause
