@echo off
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Error: Visual Studio Installer could not be found.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo Error: Visual Studio with C++ tools could not be found.
    exit /b 1
)

call "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b %errorlevel%

set "CMAKE=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "NVCC=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"

if not exist "%NVCC%" (
    echo Error: CUDA Toolkit 13.3 could not be found.
    exit /b 1
)

pushd "%~dp0"

if not exist build\build.ninja (
    "%CMAKE%" -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_CUDA_COMPILER="%NVCC%" -DCMAKE_BUILD_TYPE=Release
    if errorlevel 1 exit /b %errorlevel%
)

"%CMAKE%" --build build
if errorlevel 1 exit /b %errorlevel%

build\optix_triangle.exe optix_triangle.ppm
set "RESULT=%errorlevel%"

popd
exit /b %RESULT%

