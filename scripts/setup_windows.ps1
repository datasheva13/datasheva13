# One-time setup on Windows. Run from the project folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Write-Host "`n== 1/5 Python ==" -ForegroundColor Cyan
if (-not (Have "python")) {
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    Write-Host "Python installed. Close this window, open a new PowerShell, and run this script again." -ForegroundColor Yellow
    exit 0
}
python --version

Write-Host "`n== 2/5 FFmpeg (with NVIDIA GPU encoding) ==" -ForegroundColor Cyan
if (-not (Have "ffmpeg")) {
    winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    Write-Host "FFmpeg installed. Close this window, open a new PowerShell, and run this script again." -ForegroundColor Yellow
    exit 0
}
ffmpeg -hide_banner -version | Select-Object -First 1

Write-Host "`n== 3/5 Python packages ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip -q
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -q

Write-Host "`n== 4/5 Thumbnail font (Cinzel, free Google font) ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path fonts, secrets, inbox, handoff, music | Out-Null
if (-not (Test-Path "fonts\Cinzel-Regular.ttf")) {
    Invoke-WebRequest "https://github.com/google/fonts/raw/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf" -OutFile "fonts\Cinzel-Regular.ttf"
}

Write-Host "`n== 5/5 Config files ==" -ForegroundColor Cyan
if (-not (Test-Path "config.yaml")) { Copy-Item config.example.yaml config.yaml }
if (-not (Test-Path ".env")) { Copy-Item .env.example .env }
Write-Host "Created config.yaml and .env (edit .env to add your keys - see README)."

Write-Host "`n== Check ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m moto_shorts check
Write-Host "`nSetup done. Next: README step 3 (keys) and step 5 (daily schedule)." -ForegroundColor Green
