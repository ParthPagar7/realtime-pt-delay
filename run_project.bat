@echo off
title Real-Time Transport Delay Predictor
color 0b

echo ===============================================
echo   🚀 Starting Persevex Transport Delay Predictor
echo ===============================================
echo.

cd /d "C:\Users\parth\Documents\realtime-pt-delay"

REM Activate virtual environment
call .\venv\Scripts\activate

REM Optional: train model first (uncomment if needed)
REM python train.py

REM Run Streamlit app
echo Launching Streamlit...
streamlit run app.py

pause
