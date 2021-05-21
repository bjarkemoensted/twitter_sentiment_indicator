@echo off
echo Setting up virtual environtment...
python -m venv env
echo Activating venv
CALL env\Scripts\activate
echo Installing required packages from requirements.txt. This may take a while.
pip install -r requirements.txt --trusted-host pipy.python.org --trusted-host files.pythonhosted.org --default-timeout=1000
CALL deactivate
echo All done!