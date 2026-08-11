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

pushd "%ROOT%"
"%XELATEX%" -interaction=nonstopmode -halt-on-error -output-directory="%BUILD_OUT%" report_cn_lrat.tex
if errorlevel 1 goto :fail
"%XELATEX%" -interaction=nonstopmode -halt-on-error -output-directory="%BUILD_OUT%" report_cn_lrat.tex
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\report_cn_lrat.pdf" "%OUT%\report_cn_lrat.pdf" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\report_cn_lrat.pdf" "%ROOT%\report_cn_lrat.pdf" >nul
if errorlevel 1 goto :fail
popd
echo PDF written to "%OUT%\report_cn_lrat.pdf"
echo Release copy written to "%ROOT%\report_cn_lrat.pdf"
popd
exit /b 0

:fail
echo XeLaTeX failed; see "%BUILD_OUT%\report_cn_lrat.log"
popd
popd
exit /b 1
