@echo off
cd /d "%~dp0\.."
title Labeling
if "%~1"=="" (
  echo Usage: tool\labeling.bat sessions\<session_dir>
  exit /b 1
)
python tool\labeling_ui.py %*
