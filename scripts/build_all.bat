@echo off
:: ============ 参数化路径（按本机修改这三个变量） ============
set "VSVCVARS=D:\VS2022BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH_LOCAL=D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
set "OPENCV_INSTALL=D:\opencv-cuda-build\install"
:: ===========================================================

if not exist "%VSVCVARS%" (
    echo 找不到 vcvars64.bat，请修改本脚本 VSVCVARS 变量
    exit /b 1
)
call "%VSVCVARS%" >nul
cd /d "%~dp0.."
if not exist build\obj mkdir build\obj

for %%f in (src\integral.cu src\mean_filter.cu src\morphology.cu src\gaussian.cu src\histogram.cu) do (
  echo === %%f
  nvcc -O3 --use_fast_math -std=c++17 -arch=sm_61 -Xcompiler "/utf-8 /W3" -c %%f -o build\obj\%%~nf.obj || goto :fail
)
echo COMPILE_ALL_OK
set CUDAINC="%CUDA_PATH_LOCAL%\include"
set CVINC="%OPENCV_INSTALL%\include"
set CVLIB="%OPENCV_INSTALL%\x64\vc17\lib"
:: OpenCV 库名带版本号 4130（4.13.0），其他版本需相应修改
set LIBS=opencv_core4130.lib opencv_imgproc4130.lib opencv_cudaarithm4130.lib opencv_cudafilters4130.lib opencv_cudaimgproc4130.lib opencv_cudawarping4130.lib cudart_static.lib

nvcc -O3 --use_fast_math -std=c++17 -arch=sm_61 -Xcompiler "/utf-8 /W3" ^
  -I%CUDAINC% -I%CVINC% -Isrc ^
  bench\smoke.cu build\obj\integral.obj build\obj\mean_filter.obj build\obj\morphology.obj build\obj\gaussian.obj build\obj\histogram.obj ^
  -L%CVLIB% %LIBS% -o build\smoke.exe || goto :fail
echo LINK_SMOKE_OK

nvcc -O3 --use_fast_math -std=c++17 -arch=sm_61 -Xcompiler "/utf-8 /W3" ^
  -I%CUDAINC% -I%CVINC% -Isrc ^
  bench\benchmark.cu build\obj\integral.obj build\obj\mean_filter.obj build\obj\morphology.obj build\obj\gaussian.obj build\obj\histogram.obj ^
  -L%CVLIB% %LIBS% -o build\benchmark.exe || goto :fail
echo LINK_BENCH_OK
exit /b 0
:fail
echo BUILD_FAIL
exit /b 1
