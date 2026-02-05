$ErrorActionPreference = "Stop"

Write-Host "🚀 STARTING ML PIPELINE EXECUTION" -ForegroundColor Green
Write-Host "=================================================="

Write-Host "`n[1/4] Downloading Historical Data (Binance)..." -ForegroundColor Cyan
py historical_data_downloader.py

Write-Host "`n[2/4] Converting to Training Format..." -ForegroundColor Cyan
py convert_historical.py

Write-Host "`n[3/4] Training Models (All Symbols)..." -ForegroundColor Cyan
py train_models.py --all --epochs 50

Write-Host "`n[4/4] Running Backtest Verification..." -ForegroundColor Cyan
py backtest.py

Write-Host "`n✅ PIPELINE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "You can now run the bot with: py main.py"
