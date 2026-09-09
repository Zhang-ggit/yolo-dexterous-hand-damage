@echo off
setlocal
cd /d "%~dp0"
rem Percent signs in URLs must be doubled in a .bat source file.
set "torch_url=https://download.pytorch.org/whl/cu121/torch-2.5.1%%2Bcu121-cp310-cp310-win_amd64.whl"
set "vision_url=https://download.pytorch.org/whl/cu121/torchvision-0.20.1%%2Bcu121-cp310-cp310-win_amd64.whl"
if "%~1"=="--check-urls" (
    echo %torch_url%
    echo %vision_url%
    exit /b 0
)
where curl.exe >nul 2>&1
if errorlevel 1 (
    echo curl.exe is required. Use a current Windows 10/11 installation.
    exit /b 1
)
if not exist "wheelhouse" mkdir "wheelhouse"
if errorlevel 1 exit /b 1
call :download "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
if errorlevel 1 exit /b 1
call :download "torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl"
if errorlevel 1 exit /b 1
echo PyTorch wheels are ready in wheelhouse.
exit /b 0

:download
if exist "wheelhouse\%~1" (
    echo Reusing %~1
    exit /b 0
)
echo Downloading %~1 - partial downloads are kept for the next run.
set "download_url=%torch_url%"
if "%~1"=="torchvision-0.20.1+cu121-cp310-cp310-win_amd64.whl" set "download_url=%vision_url%"
curl.exe --location --fail --continue-at - --retry 8 --retry-delay 5 --connect-timeout 30 --speed-limit 1024 --speed-time 60 --output "wheelhouse\%~1.part" "%download_url%"
if errorlevel 1 (
    echo Download interrupted. Run this script again to resume.
    exit /b 1
)
move /y "wheelhouse\%~1.part" "wheelhouse\%~1" >nul
if errorlevel 1 exit /b 1
exit /b 0
