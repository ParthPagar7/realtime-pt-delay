import os
import subprocess
import webbrowser
import time

# Path to your venv activate script
venv_path = r"C:\Users\parth\Documents\realtime-pt-delay\venv\Scripts\activate"

# Step 1: Activate virtual environment
os.system(f'cmd /k "call {venv_path} & streamlit run app.py"')

# Step 2: Wait a bit then open in browser
time.sleep(3)
webbrowser.open("http://localhost:8501")
