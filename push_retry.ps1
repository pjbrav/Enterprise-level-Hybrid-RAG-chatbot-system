$image = "ghcr.io/pjbrav/enterprise-level-hybrid-rag-chatbot-system:latest"

for ($i = 1; $i -le 6000; $i++) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "`n=== Push attempt $i started at $ts ===" -ForegroundColor Cyan

    docker push $image 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nPUSH SUCCEEDED on attempt $i at $ts" -ForegroundColor Green
        break
    }
    Write-Host "Attempt $i failed - retrying in 30s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
}
