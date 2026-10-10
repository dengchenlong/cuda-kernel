@echo off
:: 构建 OpenCV-CUDA 基线（用于 benchmark 对照）
:: ============ 参数化路径（按本机修改） ============
set "VSVCVARS=D:\VS2022BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "OPENCV_SRC=G:\opencv"                rem OpenCV 源码
set "OPENCV_CONTRIB=G:\opencv_contrib"    rem opencv_contrib 源码（需要 cudev 模块）
set "BUILD_DIR=D:\opencv-cuda-build"
:: =================================================

if not exist "%VSVCVARS%" (
    echo 找不到 vcvars64.bat，请修改本脚本 VSVCVARS 变量
    exit /b 1
)
call "%VSVCVARS%" >nul
set PATH=D:\VS2022BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%

cmake -G Ninja -S "%OPENCV_SRC%" -B "%BUILD_DIR%" ^
  -DCMAKE_INSTALL_PREFIX=%BUILD_DIR%\install ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_LIST=core,imgproc,cudev,cudaarithm,cudawarping,cudaimgproc,cudafilters ^
  -DOPENCV_EXTRA_MODULES_PATH=%OPENCV_CONTRIB%\modules ^
  -DWITH_CUDA=ON -DCUDA_ARCH_BIN=6.1 -DCUDA_ARCH_PTX= -DWITH_NPP=ON -DWITH_CUBLAS=OFF -DWITH_CUFFT=OFF ^
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF ^
  -DWITH_IPP=OFF -DWITH_FFMPEG=OFF -DWITH_OPENCL=OFF -DWITH_ITT=OFF ^
  -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON ^
  -DBUILD_opencv_java=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=OFF ^
  -DOPENCV_GENERATE_PKGCONFIG=OFF -DOPENCV_ENABLE_NONFREE=OFF
echo CMAKE_CONFIG_EXIT=%ERRORLEVEL%
cmake --build "%BUILD_DIR%" --target install -- -j 4 2>&1
echo CMAKE_BUILD_EXIT=%ERRORLEVEL%
