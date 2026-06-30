@echo off

echo ==========================================
echo Airport Detection System Installer
echo ==========================================

call drdo_env\Scripts\activate

pip install --upgrade pip

pip install --no-index --find-links=wheelhouse -r requirements.txt

echo.
echo Installation Complete.
pause