@echo off
cd /d "%~dp0\.."
title Realtime Suppression - F10 quit, F9 bypass
python toolt_suppress.py %*
pause
