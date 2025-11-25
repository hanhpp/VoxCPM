# PowerShell script to check Docker status
Write-Host "Checking Docker status..." -ForegroundColor Cyan

# Check if Docker Desktop process is running
$dockerProcess = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
if ($dockerProcess) {
    Write-Host "✓ Docker Desktop process is running" -ForegroundColor Green
} else {
    Write-Host "✗ Docker Desktop process is NOT running" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and wait for it to fully initialize" -ForegroundColor Yellow
}

# Test Docker connection
Write-Host "`nTesting Docker connection..." -ForegroundColor Cyan
try {
    $dockerVersion = docker version --format '{{.Server.Version}}' 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker is accessible (Server version: $dockerVersion)" -ForegroundColor Green
    } else {
        Write-Host "✗ Docker daemon is not accessible" -ForegroundColor Red
        Write-Host "  Error: $dockerVersion" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Failed to connect to Docker daemon" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Yellow
}

# Test Docker Compose
Write-Host "`nTesting Docker Compose..." -ForegroundColor Cyan
try {
    $composeVersion = docker compose version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker Compose is working" -ForegroundColor Green
        Write-Host "  $composeVersion" -ForegroundColor Gray
    } else {
        Write-Host "✗ Docker Compose error" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Docker Compose not available" -ForegroundColor Red
}

Write-Host "`nIf Docker is not accessible, try:" -ForegroundColor Yellow
Write-Host "  1. Restart Docker Desktop" -ForegroundColor White
Write-Host "  2. Run Docker Desktop as Administrator" -ForegroundColor White
Write-Host "  3. Check Windows Services for 'com.docker.service'" -ForegroundColor White

