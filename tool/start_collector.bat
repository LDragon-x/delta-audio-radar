@echo off
cd /d "%~dp0\.."
title Delta Collector - F10 stop
python tool\collector.py %*
