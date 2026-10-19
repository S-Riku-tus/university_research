[CmdletBinding()]
param([string]$EnginePath = '')

$ErrorActionPreference = 'Stop'
$thesisDirectory = $PSScriptRoot
$thesisOutput = Join-Path $thesisDirectory 'build'
New-Item -ItemType Directory -Force -Path $thesisOutput | Out-Null

if (-not $EnginePath) {
    $thesisEngineCommand = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($thesisEngineCommand) { $EnginePath = $thesisEngineCommand.Source }
}
if (-not $EnginePath) {
    $thesisTools = Join-Path $thesisDirectory '.tools/tectonic-0.17.0'
    $EnginePath = Join-Path $thesisTools 'tectonic.exe'
    if (-not (Test-Path -LiteralPath $EnginePath)) {
        New-Item -ItemType Directory -Force -Path $thesisTools | Out-Null
        $thesisArchive = Join-Path $thesisTools 'tectonic.zip'
        $thesisUrl = 'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip'
        Invoke-WebRequest -UseBasicParsing -Uri $thesisUrl -OutFile $thesisArchive
        $thesisExpectedHash = 'f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f'
        $thesisActualHash = (Get-FileHash -LiteralPath $thesisArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($thesisActualHash -ne $thesisExpectedHash) { throw 'Tectonic archive checksum mismatch.' }
        Expand-Archive -LiteralPath $thesisArchive -DestinationPath $thesisTools -Force
    }
}
if (-not (Test-Path -LiteralPath $EnginePath)) { throw "TeX engine not found: $EnginePath" }

# The override exists only for this build process. No profile or global PATH is edited.
$thesisPreviousCache = $env:TECTONIC_CACHE_DIR
$env:TECTONIC_CACHE_DIR = Join-Path $thesisDirectory '.cache/tectonic'
try {
    Push-Location -LiteralPath $thesisDirectory
    try {
        & $EnginePath --keep-logs --keep-intermediates --synctex --outdir $thesisOutput main.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX failed; inspect build/main.log." }
    }
    finally { Pop-Location }
}
finally {
    if ($null -eq $thesisPreviousCache) { Remove-Item Env:TECTONIC_CACHE_DIR -ErrorAction SilentlyContinue }
    else { $env:TECTONIC_CACHE_DIR = $thesisPreviousCache }
}
Write-Output "PDF: $(Join-Path $thesisOutput 'main.pdf')"
