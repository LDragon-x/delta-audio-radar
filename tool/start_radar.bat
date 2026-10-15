@echo off
cd /d "%~dp0\.."
title Delta Radar
python main.py --loopback
pause
