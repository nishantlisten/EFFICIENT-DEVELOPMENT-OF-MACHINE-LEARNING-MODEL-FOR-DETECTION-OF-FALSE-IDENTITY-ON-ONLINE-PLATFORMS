@echo off
echo ============================================
echo  Starting Fake Profile Detection Server
echo ============================================

:: Kill any existing Python server
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul

:: Change to backend directory
cd /d d:\Downloads\fake_profile\fake_profile\backend

:: Run the Django server
echo Starting Django server on http://127.0.0.1:8000 ...
d:\Downloads\fake_profile\fake_profile\venv_313\Scripts\python.exe manage.py runserver 8000

pause
