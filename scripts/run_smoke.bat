@echo off
:: ============ 参数化路径（按本机修改） ============
set "CUDA_BIN=D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
set "OPENCV_BIN=D:\opencv-cuda-build\install\x64\vc17\bin"
:: =================================================
set PATH=%OPENCV_BIN%;%CUDA_BIN%;%PATH%
cd /d "%~dp0..\build"
smoke.exe 2>&1
echo SMOKE_EXIT=%ERRORLEVEL%
