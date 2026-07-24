Write-Host "Starting Keycloak Server on Port 8081..." -ForegroundColor Green
Write-Host "Auto-importing 'multidimensions' realm and 'rag-sales-bot' client..." -ForegroundColor Cyan

$env:KEYCLOAK_ADMIN="admin"
$env:KEYCLOAK_ADMIN_PASSWORD="admin"

cd "..\keycloak\keycloak-25.0.2\bin"
.\kc.bat start-dev --http-port 8081 --import-realm
