@echo off

REM Activate virtual environment - adjust path if needed
call venv\Scripts\activate

REM Run unified python launcher script that starts everything
python run_all.py

pause
