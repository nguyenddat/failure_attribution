$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Join-Path $workspaceRoot "who&when"

if (!(Test-Path (Join-Path $projectRoot "main.py"))) {
    throw "Cannot find main.py at $projectRoot"
}

$pm2Name = "ww-step-by-step"

# Ensure pm2 is available
$pm2Cmd = Get-Command pm2 -ErrorAction SilentlyContinue
if (-not $pm2Cmd) {
    throw "pm2 is not installed or not in PATH. Install with: npm i -g pm2"
}

Set-Location $projectRoot

# Delete old process if exists to avoid duplicate-name conflicts
pm2 delete $pm2Name | Out-Null

# Start with pm2 so process survives VSCode close
pm2 start python --name $pm2Name -- main.py

Write-Host ""
Write-Host "Started with pm2."
Write-Host "Name: $pm2Name"
Write-Host "Project: $projectRoot"
Write-Host ""
Write-Host "Status: pm2 status"
Write-Host "Logs:   pm2 logs $pm2Name"
Write-Host "Stop:   pm2 stop $pm2Name"
Write-Host "Delete: pm2 delete $pm2Name"
