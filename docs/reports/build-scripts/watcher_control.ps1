[CmdletBinding()]
param(
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'

$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Watcher = [IO.Path]::GetFullPath((Join-Path $ScriptsDir 'watch_windows.ps1'))
$WindowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

function Get-ReportWatcher {
    Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine.IndexOf($Watcher, [StringComparison]::OrdinalIgnoreCase) -ge 0
        }
}

function Get-ReportWatcherCount {
    return @((Get-ReportWatcher)).Count
}

switch ($Action) {
    'status' {
        $running = @(Get-ReportWatcher)
        if ($running.Count -eq 0) {
            Write-Host 'report watcher: stopped'
        }
        else {
            $running | ForEach-Object {
                Write-Host ('report watcher: running, PID {0}' -f $_.ProcessId)
            }
        }
        exit 0
    }

    'start' {
        $running = @(Get-ReportWatcher)
        if ($running.Count -gt 0) {
            $running | ForEach-Object {
                Write-Host ('report watcher already running, PID {0}' -f $_.ProcessId)
            }
            exit 0
        }

        if (-not (Test-Path -LiteralPath $Watcher)) {
            throw "Cannot find watcher: $Watcher"
        }
        if (-not (Test-Path -LiteralPath $WindowsPowerShell)) {
            throw "Cannot find Windows PowerShell: $WindowsPowerShell"
        }

        Start-Process -FilePath $WindowsPowerShell `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Watcher) `
            -WorkingDirectory $env:SystemRoot `
            -WindowStyle Minimized | Out-Null

        Start-Sleep -Milliseconds 700
        $running = @(Get-ReportWatcher)
        if ($running.Count -eq 0) {
            throw 'The report watcher did not remain running.'
        }
        $running | ForEach-Object {
            Write-Host ('report watcher started, PID {0}' -f $_.ProcessId)
        }
        exit 0
    }

    'stop' {
        $running = @(Get-ReportWatcher)
        if ($running.Count -eq 0) {
            Write-Host 'report watcher already stopped'
            exit 0
        }

        $running | ForEach-Object {
            Write-Host ('stopping report watcher, PID {0}' -f $_.ProcessId)
            Stop-Process -Id ([int]$_.ProcessId) -Force
        }

        Start-Sleep -Milliseconds 500
        if ((Get-ReportWatcherCount) -ne 0) {
            throw 'The report watcher could not be stopped.'
        }
        Write-Host 'report watcher stopped'
        exit 0
    }
}
