@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>&1
if errorlevel 1 (
    echo Please run this file from Anaconda Prompt.
    exit /b 1
)
call conda run -n hand-yolo26-win python --version >nul 2>&1
if not errorlevel 1 goto :install
echo Creating a separate environment: hand-yolo26-win
call conda create -n hand-yolo26-win python=3.10 -y
if errorlevel 1 goto :failed
:install
call download_windows.bat
if errorlevel 1 goto :failed
call conda run --no-capture-output -n hand-yolo26-win python -m pip install --timeout 120 --retries 10 "wheelhouse\torch-2.5.1+cu121-cp310-cp310-win_amd64.whl" "wheelhouse\torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl"
if errorlevel 1 goto :failed
call conda run --no-capture-output -n hand-yolo26-win python -m pip install --timeout 120 --retries 10 -r requirements.txt
if errorlevel 1 goto :failed
call conda run --no-capture-output -n hand-yolo26-win python -m pip check
if errorlevel 1 goto :failed
echo Installation complete. Run test_windows.bat next.
exit /b 0
:failed
echo Installation stopped. Please share the error above.
exit /b 1
