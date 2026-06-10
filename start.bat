@echo off

cd /d "%~dp0"

start "" ollama serve

timeout /t 5 /nobreak > nul

start "" cmd /c "venv\Scripts\activate && python api.py"

timeout /t 3 /nobreak > nul

start "" cmd /c "venv\Scripts\activate && python ui.py"

timeout /t 5 /nobreak > nul

start http://127.0.0.1:7860