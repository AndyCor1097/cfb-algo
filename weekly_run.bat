@echo off
echo ==========================================
echo  TheCFBAlgo -- Weekly Update
echo ==========================================
cd /d "%~dp0"

set /p WEEK="Enter week number to predict: "
set /p LASTWEEK="Enter last week number to grade: "

echo.
echo [1/6] Fetching in-season 2026 stats...
python fetch_inseason.py

echo.
echo [2/6] Fetching latest lines from Odds API...
python fetch_lines.py

echo.
echo [3/6] Grading last week...
python grade.py %LASTWEEK%

echo.
echo [4/6] Rebuilding features...
python features.py

echo.
echo [5/6] Retraining model...
python model.py

echo.
echo [6/6] Generating predictions + pushing to site...
python predict.py %WEEK% --push

echo.
echo [+] Pushing tracker...
git add tracker_summary.json
git commit -m "Update tracker week %LASTWEEK%"
git push

echo.
echo Done! Site updated at thecfbalgo.streamlit.app
pause
