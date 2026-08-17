@echo off
echo Creating Python Virtual Environment...
python -m venv venv

echo Activating Environment...
:: You MUST use "call" here, otherwise the script will stop after activating
call venv\Scripts\activate

echo Installing Requirements...
pip install -r requirements.txt

echo Check Update Python...
python.exe -m pip install --upgrade pip

echo Setup complete! Your environment is now ready.
:: cmd /k keeps the window open so you can start working in the active environment
cmd /k