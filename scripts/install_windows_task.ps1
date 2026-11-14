# Schedules the daily run with Windows Task Scheduler (wakes the laptop if it's asleep).
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows_task.ps1 [-Time 06:15]
param([string]$Time = "06:15")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$bat = Join-Path $root "scripts\run_daily.bat"

$action   = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask -TaskName "MotoShorts Daily" -Action $action -Trigger $trigger -Settings $settings `
            -Description "Makes today's 3 moto shorts for review" -Force | Out-Null
Write-Host "Scheduled 'MotoShorts Daily' every day at $Time." -ForegroundColor Green
Write-Host "Tip: also enable wake timers: Control Panel > Power Options > Change plan settings >"
Write-Host "     Advanced > Sleep > Allow wake timers = Enable (plugged in)."
Write-Host "To remove: Unregister-ScheduledTask -TaskName 'MotoShorts Daily'"
