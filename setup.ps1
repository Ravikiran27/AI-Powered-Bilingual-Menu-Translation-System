# Quick Setup and Run Script for Windows PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Bilingual Menu Translation System" -ForegroundColor Cyan
Write-Host "  Setup & Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Create virtual environment (optional but recommended)
Write-Host "`nDo you want to create a virtual environment? (y/n): " -ForegroundColor Yellow -NoNewline
$createVenv = Read-Host

if ($createVenv -eq 'y' -or $createVenv -eq 'Y') {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
    
    Write-Host "✓ Virtual environment created and activated" -ForegroundColor Green
}

# Install requirements
Write-Host "`nInstalling required packages..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray

pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All packages installed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Error installing packages" -ForegroundColor Red
    exit 1
}

# Check if models exist
Write-Host "`nChecking for trained models..." -ForegroundColor Yellow
if (Test-Path "saved_model/en_hi" -and Test-Path "saved_model/en_kn") {
    Write-Host "✓ Trained models found!" -ForegroundColor Green
    $trainModels = $false
} else {
    Write-Host "! Models not found. You need to train them first." -ForegroundColor Yellow
    $trainModels = $true
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Show next steps
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host ""

if ($trainModels) {
    Write-Host "1. Train the models by running the Jupyter notebook:" -ForegroundColor White
    Write-Host "   jupyter notebook bilingual_menu_translation.ipynb" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "2. After training, run the web app:" -ForegroundColor White
    Write-Host "   For Streamlit: streamlit run app.py" -ForegroundColor Cyan
    Write-Host "   For Gradio:    python app_gradio.py" -ForegroundColor Cyan
} else {
    Write-Host "Run the web application:" -ForegroundColor White
    Write-Host "  For Streamlit: streamlit run app.py" -ForegroundColor Cyan
    Write-Host "  For Gradio:    python app_gradio.py" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Gray
Write-Host ""

# Ask if user wants to launch Jupyter now
if ($trainModels) {
    Write-Host "Would you like to launch Jupyter Notebook now? (y/n): " -ForegroundColor Yellow -NoNewline
    $launchJupyter = Read-Host
    
    if ($launchJupyter -eq 'y' -or $launchJupyter -eq 'Y') {
        Write-Host "`nLaunching Jupyter Notebook..." -ForegroundColor Green
        jupyter notebook bilingual_menu_translation.ipynb
    }
}
