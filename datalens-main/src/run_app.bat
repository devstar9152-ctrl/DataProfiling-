@echo off
:: --------------------------------------------
:: Data Profiling AI Assistant - Launch Script
:: --------------------------------------------
:: Step 1: Navigate to project folder
cd /d "C:\Users\sound\OneDrive\Desktop\DataProfilingAI"

:: Step 2: Activate virtual environment
call .venv\Scripts\activate

:: Step 3: Launch Streamlit app
echo Starting Data Profiling AI Assistant...
python -m streamlit run app.py

:: Step 4: Keep window open after closing Streamlit
echo.
echo To stop the app, press CTRL + C in this window.
pause
