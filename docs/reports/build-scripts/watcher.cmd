@echo off
setlocal
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
set "CONTROL=%~dp0watcher_control.ps1"

if "%~1"=="" set "ACTION=status"
if not "%~1"=="" set "ACTION=%~1"

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%CONTROL%" %ACTION%
exit /b %ERRORLEVEL%
