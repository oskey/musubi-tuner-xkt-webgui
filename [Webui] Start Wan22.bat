@echo off
chcp 65001 > nul

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
:: 2. 激活虚拟环境
:: -------------------------------
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [错误] 虚拟环境激活失败，请手动执行：venv\Scripts\activate.bat
    pause
    exit /b
)
echo ✅ 虚拟环境已激活



:: -------------------------------
:: 3. 运行Wan22
:: -------------------------------
echo.
echo =================================================
echo [安装] 开始运行Wan2.2训练工具
echo =================================================
echo ⚠️ Wan2.2训练我这里是直接同时训练高低噪声，效果更佳，但显存占用较大！
echo ⚠️ 显存优化可以参考.md文档，进行调整
echo ******************************************************************************

python wan22_webui.py

pause
