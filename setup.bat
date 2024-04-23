@echo off

REM Install zenml
echo INFO: Installing zenml server for Python
pip install zenml[server]

REM Checking if zenml is installed correctly
zenml -v

if %errorlevel% NEQ 0 (
    echo ERROR: Installing zenml failed...
    echo FAILED COMMAND: pip install zenml[server] or zenml -v
    goto :FAIL
)
echo INFO: Successfully installed zenml

REM Install sklearn for zenml
echo INFO: Installing sklearn for zenml
zenml integration install sklearn -y

if %errorlevel% NEQ 0 (
    echo ERROR: Installing sklearn inside zenml failed...
    echo FAILED COMMAND: zenml integration install sklearn -y
    goto :FAIL
)
echo INFO: Successfully installed sklearn for zenml

REM Install mlflow for zenml
echo INFO: Installing mlflow for zenml
zenml integration install mlflow -y

if %errorlevel% NEQ 0 (
    echo ERROR: Installing mlflow inside zenml failed...
    echo FAILED COMMAND: zenml integration install mlflow -y
    goto :FAIL
)
echo INFO: Successfully installed mlflow for zenml


goto :SUCCESS
exit /b


:SUCCESS
echo INFO: Dependencies installed successfully!
exit /b 0

:FAIL
echo ERROR: Error while installing dependencies!
exit /b 1