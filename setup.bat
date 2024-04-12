@echo off

REM Install zenml
echo INFO: Installing zenml server for Python
pip install zenml[server]

REM Checking if zenml is installed correctly
zenml -v

if %errorlevel% NEQ 0 (
    echo ERROR: Installing zenml failed...
    echo FAILED COMMAND: pip install zenml[server]
    call :FAIL
)
echo INFO: Successfully installed zenml

REM Install sklearn for zenml
echo INFO: Installing sklearn for zenml
zenml integration install sklearn -y

if %errorlevel% NEQ 0 (
    echo ERROR: Installing sklearn inside zenml failed...
    echo FAILED COMMAND: zenml integration install sklearn -y
    call :FAIL
)
echo INFO: Successfully installed sklearn for zenml


call :SUCCESS
exit /b


:SUCCESS
echo INFO: Dependencies installed successfully!
exit /b 0

:FAIL
echo ERROR: Error while installing dependencies!
exit /b 1