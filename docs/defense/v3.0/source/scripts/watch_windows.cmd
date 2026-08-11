@echo off
setlocal EnableExtensions

rem Double-click this file to rebuild v3.0 after every TeX/style save.
rem Keep the window open; press Ctrl+C to stop.

set "WATCHER=%~dp0watch_windows.ps1"
if not exist "%WATCHER%" (
  echo Cannot find "%WATCHER%".
  exit /b 2
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%WATCHER%"
set "STATUS=%ERRORLEVEL%"
if not "%STATUS%"=="0" (
  echo Watcher exited with code %STATUS%.
)
exit /b %STATUS%
