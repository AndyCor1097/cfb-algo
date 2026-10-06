@echo off
set /p WEEK="Enter week number: "
echo Running predictions for Week %WEEK%...
cd /d "%~dp0"
python predict.py %WEEK%
pause
