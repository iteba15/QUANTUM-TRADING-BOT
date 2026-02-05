@echo off
echo ===================================================
echo   QUANTUM BOT COMPILER
echo ===================================================
echo.
echo Installing requirements...
.\.venv\Scripts\pip install -r requirements_full.txt pyinstaller

echo.
echo Cleaning previous builds...
rmdir /s /q build
rmdir /s /q dist
del /q *.spec

echo.
echo Compiling QuantumBot...
.\.venv\Scripts\pyinstaller --name QuantumBot ^
    --onefile ^
    --clean ^
    --hidden-import pandas ^
    --hidden-import numpy ^
    --hidden-import torch ^
    --hidden-import requests ^
    --add-data "TRAINING_GUIDE.md;." ^
    --add-data "models;models" ^
    --icon=NONE ^
    main.py

echo.
if exist "dist\QuantumBot.exe" (
    echo [SUCCESS] Compilation complete!
    echo Executable is located in: dist\QuantumBot.exe
) else (
    echo [ERROR] Compilation failed.
)
pause
