param(
    [int]$PollMilliseconds = 700,
    [int]$DebounceMilliseconds = 350,
    [int]$MaxBuilds = 0
)

$ErrorActionPreference = 'Stop'

$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptsDir
$SourceRoot = Join-Path $ProjectRoot 'src'
$ThemeRoot = Join-Path $ProjectRoot 'theme'
$WatchRoots = @($SourceRoot, $ThemeRoot)
$BuildScript = Join-Path $ScriptsDir 'build_windows.cmd'
$BuildCount = 0

function Get-SourceSignature {
    $files = foreach ($root in $WatchRoots) {
        Get-ChildItem -LiteralPath $root -Recurse -File |
            Where-Object { $_.Extension.ToLowerInvariant() -in @('.tex', '.sty') }
    }
    $files = $files | Sort-Object FullName

    return (($files | ForEach-Object {
        '{0}|{1}|{2}' -f $_.FullName, $_.Length, $_.LastWriteTimeUtc.Ticks
    }) -join "`n")
}

function Invoke-Build {
    param([string]$Reason)

    $started = Get-Date
    Write-Host ('[{0}] Rebuilding ({1})...' -f $started.ToString('HH:mm:ss'), $Reason) -ForegroundColor Cyan

    # Calling the existing script keeps the one-shot and watch builds identical.
    # Start-Process is used because PowerShell does not reliably execute a .cmd
    # located under a WSL UNC path through the call operator alone.
    $buildProcess = Start-Process -FilePath $env:ComSpec `
        -ArgumentList @('/d', '/c', ('call "{0}"' -f $BuildScript)) `
        -Wait -PassThru -NoNewWindow
    $status = $buildProcess.ExitCode
    $script:BuildCount++

    if ($status -eq 0) {
        Write-Host ('[{0}] Updated output/v3.0.pdf and release v3.0.pdf.' -f (Get-Date).ToString('HH:mm:ss')) -ForegroundColor Green
    }
    else {
        Write-Warning ('[{0}] Build failed with exit code {1}; watching continues.' -f (Get-Date).ToString('HH:mm:ss'), $status)
    }

    # Optional bounded mode is useful for a smoke test; normal use is unlimited.
    if ($MaxBuilds -gt 0 -and $script:BuildCount -ge $MaxBuilds) {
        exit $status
    }
}

Write-Host 'CCIR v3.0 automatic TeX build' -ForegroundColor White
Write-Host ('Sources: {0}; {1}' -f $SourceRoot, $ThemeRoot)
Write-Host 'Watching .tex and .sty files. Press Ctrl+C to stop.'
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

    # Editors may write a file in several operations; wait until the tree is stable.
    do {
        $candidateSignature = $currentSignature
        Start-Sleep -Milliseconds $DebounceMilliseconds
        $currentSignature = Get-SourceSignature
    } while ($currentSignature -ne $candidateSignature)

    Invoke-Build -Reason 'source changed'
    $lastSignature = Get-SourceSignature
}
