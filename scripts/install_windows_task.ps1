# Runs Moto Shorts every time you sign in to Windows (after start-up), once per day at most.
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows_task.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$bat = Join-Path $root "scripts\run_daily.bat"

# remove the old morning schedule if it exists
Unregister-ScheduledTask -TaskName "MotoShorts Daily" -Confirm:$false -ErrorAction SilentlyContinue

$action   = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$trigger.Delay = "PT2M"                 # wait 2 minutes so Wi-Fi is connected
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask -TaskName "MotoShorts Startup" -Action $action -Trigger $trigger -Settings $settings `
            -Description "Makes today's moto shorts for review (once per day, at sign-in)" -Force | Out-Null
Write-Host "Moto Shorts will run 2 minutes after you sign in (once per day)." -ForegroundColor Green
Write-Host "To remove: Unregister-ScheduledTask -TaskName 'MotoShorts Startup'"
