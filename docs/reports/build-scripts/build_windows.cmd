@echo off
setlocal EnableExtensions
pushd "%~dp0\.."
set "ROOT=%CD%"
set "RELEASE=%ROOT%\.."
set "BUILD_OUT=%ROOT%\.build_v30"
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

for %%D in (report_cn_2p report_en) do (
  "%XELATEX%" -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="%BUILD_OUT%" "%%D.tex"
  if errorlevel 1 goto :fail
  "%XELATEX%" -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="%BUILD_OUT%" "%%D.tex"
  if errorlevel 1 goto :fail
)

copy /Y "%BUILD_OUT%\report_cn_2p.pdf" "%ROOT%\report_cn_2p.pdf" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\report_cn_2p.pdf" "%RELEASE%\report_cn_2p.pdf" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\report_cn_2p.pdf" "%RELEASE%\DefaultGroup-362259-方案介绍.pdf" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\report_en.pdf" "%ROOT%\report_en.pdf" >nul
if errorlevel 1 goto :fail
copy /Y "%BUILD_OUT%\report_en.pdf" "%RELEASE%\report_en.pdf" >nul
if errorlevel 1 goto :fail

echo Chinese and English report PDFs updated.
echo Formal Chinese attachment: "%RELEASE%\DefaultGroup-362259-方案介绍.pdf"
popd
exit /b 0

:fail
echo XeLaTeX failed; see "%BUILD_OUT%\report_cn_2p.log" or "%BUILD_OUT%\report_en.log"
popd
exit /b 1
