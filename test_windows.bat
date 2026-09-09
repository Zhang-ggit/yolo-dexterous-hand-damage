@echo off
setlocal
cd /d "%~dp0"
where conda >nul 2>&1
if errorlevel 1 (
    echo Please run this file from Anaconda Prompt.
    exit /b 1
)
echo Checking PyTorch, NumPy and CUDA...
call conda run --no-capture-output -n hand-yolo26-win python -c "import torch,numpy as np,cv2,ultralytics; print('Torch:',torch.__version__); print('YOLO:',ultralytics.__version__); print('OpenCV:',cv2.__version__); print('CUDA:',torch.version.cuda); print('GPU available:',torch.cuda.is_available()); a=torch.from_numpy(np.ones((2,2),dtype=np.float32)); print('NumPy bridge:',a.numpy()); device='cuda' if torch.cuda.is_available() else 'cpu'; b=a.to(device); print('Compute:',(b@b).cpu().numpy()); print('Device:',torch.cuda.get_device_name(0) if device=='cuda' else 'CPU')"
if errorlevel 1 goto :failed
echo Running single-image inference with the supplied weights...
call conda run --no-capture-output -n hand-yolo26-win python predict_hand_status.py
if errorlevel 1 goto :failed
echo Windows inference test passed.
echo Optional one-epoch training test:
echo conda run --no-capture-output -n hand-yolo26-win python train_det.py --epochs 1 --batch 16 --name windows_smoke
exit /b 0
:failed
echo Test stopped. Please share the error above.
exit /b 1
