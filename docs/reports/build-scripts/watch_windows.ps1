param(
    [int]$PollMilliseconds = 700,
    [int]$DebounceMilliseconds = 350,
    [int]$MaxBuilds = 0
)

$ErrorActionPreference = 'Stop'

$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ReportsRoot = Split-Path -Parent $ScriptsDir
$BuildScript = Join-Path $ScriptsDir 'build_windows.cmd'
$WatchFiles = @(
    (Join-Path $ReportsRoot 'report_cn_2p.tex'),
    (Join-Path $ReportsRoot 'report_en.tex')
)
$BuildCount = 0

function Get-SourceSignature {
    return (($WatchFiles | ForEach-Object {
        $item = Get-Item -LiteralPath $_
        '{0}|{1}|{2}' -f $item.FullName, $item.Length, $item.LastWriteTimeUtc.Ticks
    }) -join "`n")
}

function Invoke-Build {
    param([string]$Reason)

    $started = Get-Date
    Write-Host ('[{0}] Rebuilding report PDFs ({1})...' -f $started.ToString('HH:mm:ss'), $Reason) -ForegroundColor Cyan

    # Use the same command-shell path as the v3 watcher so WSL UNC paths work.
    $buildProcess = Start-Process -FilePath $env:ComSpec `
        -ArgumentList @('/d', '/c', ('call "{0}"' -f $BuildScript)) `
        -Wait -PassThru -NoNewWindow
    $status = $buildProcess.ExitCode
    $script:BuildCount++

    if ($status -eq 0) {
        Write-Host ('[{0}] Updated report_cn_2p.pdf, DefaultGroup-362259-方案介绍.pdf, and report_en.pdf.' -f (Get-Date).ToString('HH:mm:ss')) -ForegroundColor Green
    }
    else {
        Write-Warning ('[{0}] Report build failed with exit code {1}; watching continues.' -f (Get-Date).ToString('HH:mm:ss'), $status)
    }

    if ($MaxBuilds -gt 0 -and $script:BuildCount -ge $MaxBuilds) {
        exit $status
    }
}

Write-Host 'CCIR report automatic TeX build' -ForegroundColor White
Write-Host ('Watching: {0}; {1}' -f $WatchFiles[0], $WatchFiles[1])
Write-Host 'Press Ctrl+C to stop.'
Write-Host ''

$lastSignature = Get-SourceSignature
Invoke-Build -Reason 'initial build'
$lastSignature = Get-SourceSignature

while ($true) {
    Start-Sleep -Milliseconds $PollMilliseconds
    $currentSignature = Get-SourceSignature

    if ($currentSignature -eq $lastSignature) {
        continue
    }

    do {
        $candidateSignature = $currentSignature
        Start-Sleep -Milliseconds $DebounceMilliseconds
        $currentSignature = Get-SourceSignature
    } while ($currentSignature -ne $candidateSignature)

    Invoke-Build -Reason 'source changed'
    $lastSignature = Get-SourceSignature
}
