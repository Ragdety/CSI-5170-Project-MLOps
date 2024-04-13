@echo off

REM Author: Edgar Terrazas
REM Date: May 2022

REM Make sure pip is up-to-date
echo Updating pip...
python -m pip install --upgrade pip
if %errorlevel% NEQ 0 (
    echo ERROR: Failed to update pip.
    goto :FAIL
)

REM Check pip version
echo Checking pip version...
pip --version
if %errorlevel% NEQ 0 (
    echo ERROR: Failed to check pip version.
    goto :FAIL
)

REM Install pipreqs using pip
echo Installing pipreqs...
pip install pipreqs
if %errorlevel% NEQ 0 (
    echo ERROR: Failed to install pipreqs.
    goto :FAIL
)

REM Print information about pipreqs
echo Information about pipreqs:
pip show pipreqs
if %errorlevel% NEQ 0 (
    echo ERROR: Failed to get information about pipreqs.
    goto :FAIL
)

REM Create a backup of current requirements.txt
echo Creating a backup of requirements.txt...
if exist requirements.txt (
    copy requirements.txt requirements_backup.txt > nul
    if %errorlevel% NEQ 0 (
        echo ERROR: Failed to create a backup of requirements.txt.
        goto :FAIL
    )
)

REM Delete current requirements.txt
echo Deleting current requirements.txt...
del requirements.txt > nul
if %errorlevel% NEQ 0 (
    echo ERROR: Failed to delete current requirements.txt.
    goto :FAIL
)

REM Execute pipreqs to create requirements.txt
echo Generating requirements.txt...
pipreqs . --force
if %errorlevel% NEQ 0 (
    echo ERROR: Failed to generate requirements.txt.
    goto :FAIL
)

REM Install requirements.txt
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% NEQ 0 (
    echo ERROR: Failed to install requirements.
    goto :FAIL
)

echo SUCCESS: All steps completed successfully.
exit /b 0

:FAIL
echo ERROR: An error occurred during the installation process.
exit /b 1