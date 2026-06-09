$ErrorActionPreference = "Stop"

$WorkspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvActivate = Join-Path $WorkspaceRoot ".env\Scripts\Activate.ps1"
$VenvPython = Join-Path $WorkspaceRoot ".env\Scripts\python.exe"

if (Test-Path $VenvActivate) {
	Write-Host "Activating workspace virtual environment at .env..."
	. $VenvActivate
	$PythonCommand = "python"
} elseif (Test-Path $VenvPython) {
	Write-Host "Using workspace virtual environment Python directly from .env..."
	$PythonCommand = $VenvPython
} else {
	throw "Virtual environment not found. Expected .env\Scripts\Activate.ps1 or .env\Scripts\python.exe under $WorkspaceRoot."
}

Write-Host "Installing Python dependencies..."
if ($PythonCommand -eq "python") {
	pip install -r (Join-Path $WorkspaceRoot "requirements.txt")
} else {
	& $PythonCommand -m pip install -r (Join-Path $WorkspaceRoot "requirements.txt")
}

Write-Host "Preloading ASR and translation models..."
if ($PythonCommand -eq "python") {
	python (Join-Path $WorkspaceRoot "scripts\preload_models.py")
} else {
	& $PythonCommand (Join-Path $WorkspaceRoot "scripts\preload_models.py")
}

Write-Host "Done. You can now run the app with local cache reuse."