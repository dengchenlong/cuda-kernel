@echo off
rem rebuild one TU only: scripts\rebuild_one.bat gaussian
setlocal
set "VSVCVARS=D:\VS2022BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set NAME=%1
if "%NAME%"=="" (echo usage: scripts\rebuild_one.bat ^<tu^> & exit /b 2)
cd /d "%~dp0.."
call "%VSVCVARS%" >nul
if not exist build\obj mkdir build\obj
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_61 -Xcompiler "/utf-8 /W3" -c src\%NAME%.cu -o build\obj\%NAME%.obj || exit /b 1
echo REBUILD_OK %NAME%
exit /b 0
