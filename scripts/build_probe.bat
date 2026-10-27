@echo off
:: 编译 build\<name>.cu 探针，自动补 UTF-8 BOM。
:: 用法：scripts\build_probe.bat <name>          例：scripts\build_probe.bat probe_m3v
:: 为什么存在：nvcc 对含中文注释的 .cu 要求 UTF-8 BOM；用 write_file 新建探针必丢 BOM，
::             每次手工 printf BOM 是重复劳动（2026-10-27 那轮重复了 3 次）。
setlocal
set "VSVCVARS=D:\VS2022BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH_LOCAL=D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
set "OPENCV_INSTALL=D:\opencv-cuda-build\install"
set NAME=%1
if "%NAME%"=="" (echo 用法: scripts\build_probe.bat ^<name^> & exit /b 2)
cd /d "%~dp0.."
if not exist "build\%NAME%.cu" (echo 找不到 build\%NAME%.cu & exit /b 2)

:: 补 BOM（前 3 字节非 EF BB BF 时才改写）
python -c "p='build/%NAME%.cu';b=open(p,'rb').read();print('BOM_ADDED') if b[:3]!=b'\xef\xbb\xbf' else None;open(p,'wb').write(b if b[:3]==b'\xef\xbb\xbf' else b'\xef\xbb\xbf'+b)" || (echo 补 BOM 失败：python 不可用？ & exit /b 2)

call "%VSVCVARS%" >nul
set CVINC="%OPENCV_INSTALL%\include"
set CVLIB="%OPENCV_INSTALL%\x64\vc17\lib"
set LIBS=opencv_core4130.lib opencv_imgproc4130.lib opencv_cudaarithm4130.lib opencv_cudafilters4130.lib opencv_cudaimgproc4130.lib opencv_cudawarping4130.lib cudart_static.lib
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_61 -Xcompiler "/utf-8 /W3" ^
  -I"%CUDA_PATH_LOCAL%\include" -I%CVINC% -Isrc ^
  build\%NAME%.cu build\obj\integral.obj build\obj\mean_filter.obj build\obj\morphology.obj build\obj\gaussian.obj build\obj\histogram.obj ^
  -L%CVLIB% %LIBS% -o build\%NAME%.exe || exit /b 1
echo PROBE_BUILD_OK %NAME%
exit /b 0
