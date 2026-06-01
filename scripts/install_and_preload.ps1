$ErrorActionPreference = "Stop"

Write-Host "Installing Python dependencies..."
pip install -r requirements.txt

Write-Host "Preloading ASR and translation models..."
python scripts/preload_models.py

Write-Host "Done. You can now run the app with local cache reuse."