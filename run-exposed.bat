@echo off
REM Script to run VoxCPM and expose it to local network
REM Run this as Administrator for automatic firewall configuration

set PORT=7860
set SCRIPT_DIR=%~dp0

echo === VoxCPM Local Network Setup ===
echo.

REM Check for admin (basic check)
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [*] Running as Administrator - Configuring firewall...

    REM Remove existing rule if it exists
    netsh advfirewall firewall delete rule name="VoxCPM Port %PORT%" >nul 2>&1

    REM Add firewall rule
    netsh advfirewall firewall add rule name="VoxCPM Port %PORT%" dir=in action=allow protocol=TCP localport=%PORT% >nul 2>&1

    echo [✓] Firewall rule added for port %PORT%
) else (
    echo [!] Not running as Administrator
    echo     Firewall may block connections. To fix:
    echo     1. Right-click this file -^> Run as Administrator
    echo     2. Run this script again
    echo.
)

REM Get local IP address (simplified)
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP:~1%

echo.
echo === Server Information ===
echo Local access:    http://localhost:%PORT%
if defined IP (
    echo Network access:  http://%IP%:%PORT%
    echo.
    echo Share this URL with other devices on your network!
) else (
    echo Network access:  (Could not detect IP address)
)
echo.

REM Set environment variables and run
set SERVER_NAME=0.0.0.0
set SERVER_PORT=%PORT%

echo Starting VoxCPM server...
echo Press Ctrl+C to stop
echo.

cd /d "%SCRIPT_DIR%"
python app.py

