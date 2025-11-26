# Script to run VoxCPM and expose it to local network
# Run this script as Administrator for automatic firewall configuration

$port = 7860
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== VoxCPM Local Network Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($isAdmin) {
    Write-Host "[*] Running as Administrator - Configuring firewall..." -ForegroundColor Yellow

    # Remove existing rule if it exists
    $existingRule = Get-NetFirewallRule -DisplayName "VoxCPM Port $port" -ErrorAction SilentlyContinue
    if ($existingRule) {
        Remove-NetFirewallRule -DisplayName "VoxCPM Port $port" -ErrorAction SilentlyContinue
    }

    # Add firewall rule
    New-NetFirewallRule -DisplayName "VoxCPM Port $port" `
        -Direction Inbound `
        -LocalPort $port `
        -Protocol TCP `
        -Action Allow `
        -Profile Domain,Private,Public | Out-Null

    Write-Host "[✓] Firewall rule added for port $port" -ForegroundColor Green
} else {
    Write-Host "[!] Not running as Administrator" -ForegroundColor Yellow
    Write-Host "    Firewall may block connections. To fix:" -ForegroundColor Yellow
    Write-Host "    1. Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    Write-Host "    2. Run this script again" -ForegroundColor Yellow
    Write-Host ""
}

# Get local IP address
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.IPAddress -notlike "127.*" -and
    $_.IPAddress -notlike "169.254.*"
} | Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "=== Server Information ===" -ForegroundColor Cyan
Write-Host "Local access:    http://localhost:$port" -ForegroundColor White
if ($ipAddress) {
    Write-Host "Network access:  http://$ipAddress:$port" -ForegroundColor White
    Write-Host ""
    Write-Host "Share this URL with other devices on your network!" -ForegroundColor Green
} else {
    Write-Host "Network access:  (Could not detect IP address)" -ForegroundColor Yellow
}
Write-Host ""

# Set environment variables and run
$env:SERVER_NAME = "0.0.0.0"
$env:SERVER_PORT = "$port"

Write-Host "Starting VoxCPM server..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

Set-Location $scriptPath
python app.py

