@echo off
rem 跑 build\<name>.exe 探针（PATH 补 CUDA + OpenCV 运行时）
rem 用法：scripts\run_probe.bat probe_f32f
set "CUDA_BIN=D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
set "OPENCV_BIN=D:\opencv-cuda-build\install\x64\vc17\bin"
set PATH=%OPENCV_BIN%;%CUDA_BIN%;%PATH%
cd /d "%~dp0..\build"
.\%~1.exe
echo PROBE_EXIT=%ERRORLEVEL%