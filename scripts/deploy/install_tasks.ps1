# Danalit — Windows Task Scheduler installation (run once, as your user):
#   powershell -ExecutionPolicy Bypass -File scripts\deploy\install_tasks.ps1
#
# Also complete the power checklist in this folder's README section below.

$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$py = "python"

function Install-Task($name, $command, $schedule, $modifier, $startTime) {
    schtasks /Delete /TN $name /F 2>$null
    $args = @("/Create", "/TN", $name, "/TR", "cmd /c cd /d `"$repo`" && $command",
              "/SC", $schedule, "/RL", "LIMITED", "/F")
    if ($modifier) { $args += @("/MO", $modifier) }
    if ($startTime) { $args += @("/ST", $startTime) }
    schtasks @args
}

Write-Host "Installing Danalit scheduled tasks (repo: $repo)"

# MT5 terminal at logon — EDIT this path to your broker's terminal64.exe:
$mt5 = "C:\Program Files\MetaTrader 5\terminal64.exe"
schtasks /Delete /TN "Danalit MT5 Terminal" /F 2>$null
schtasks /Create /TN "Danalit MT5 Terminal" /TR "`"$mt5`"" /SC ONLOGON /RL LIMITED /F

Install-Task "Danalit Collector"     "$py scripts\run_collector.py"          "ONLOGON" $null $null
Install-Task "Danalit Orchestrator"  "$py scripts\run_trading.py"            "ONLOGON" $null $null
Install-Task "Danalit Watchdog"      "$py scripts\deploy\watchdog.py"        "MINUTE"  "5"   $null
Install-Task "Danalit Daily Digest"  "$py scripts\send_digest.py"            "DAILY"   $null "21:00"
Install-Task "Danalit Retrain"       "$py scripts\run_retrain.py"            "MONTHLY" $null "02:00"
Install-Task "Danalit Probation"     "$py scripts\run_retrain.py --probation-check" "DAILY" $null "03:00"
# Optional dashboard at logon:
# Install-Task "Danalit Dashboard"   "$py scripts\run_dashboard.py"          "ONLOGON" $null $null

Write-Host ""
Write-Host "POWER / OS CHECKLIST (do these once, manually):"
Write-Host "  1. Settings > Power: never sleep, never hibernate (lid close: do nothing)"
Write-Host "  2. Windows Update > Advanced: defer restarts / set active hours 00:00-23:00"
Write-Host "  3. Disable 'Fast startup' (it breaks scheduled tasks after reboot)"
Write-Host "  4. Verify: reboot, then 'schtasks /Query /TN \"Danalit Orchestrator\"'"
