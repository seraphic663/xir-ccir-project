@echo off
setlocal EnableExtensions
pushd "%~dp0\.."
set "ROOT=%CD%"
set "OUT=%ROOT%\output"
set "BUILD_OUT=%OUT%\.build"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%BUILD_OUT%" mkdir "%BUILD_OUT%"

set "XELATEX=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
if not exist "%XELATEX%" (
  where xelatex >nul 2>nul
  if errorlevel 1 (
    echo Cannot find MiKTeX xelatex.exe.
    popd
    exit /b 2
  )
  set "XELATEX=xelatex"
)

pushd "%ROOT%\src"
"%XELATEX%" -interaction=nonstopmode -halt-on-error -output-directory="%BUILD_OUT%" v3.0.tex
if errorlevel 1 goto :fail
"%XELATEX%" -interaction=nonstopmode -halt-on-error -output-directory="%BUILD_OUT%" v3.0.tex
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\v3.0.pdf" "%OUT%\v3.0.pdf" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\v3.0.log" "%OUT%\v3.0.log" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\v3.0.pdf" "%ROOT%\v3.0.pdf" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\v3.0.pdf" "%ROOT%\..\v3.0.pdf" >nul
if errorlevel 1 goto :fail
popd
echo PDF written to "%OUT%\v3.0.pdf"
echo Release copy written to "%ROOT%\v3.0.pdf"
echo Canonical copy written to "%ROOT%\..\v3.0.pdf"
popd
exit /b 0

:fail
echo XeLaTeX failed; see "%BUILD_OUT%\v3.0.log"
popd
popd
exit /b 1
