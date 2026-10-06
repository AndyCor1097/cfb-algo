@echo off
echo ==========================================
echo  TheCFBAlgo — Starting App
echo ==========================================
cd /d "%~dp0"
streamlit run app.py --server.port 8501
pause
