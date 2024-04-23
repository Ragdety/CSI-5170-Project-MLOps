@echo off

set projectPath=../../../
set runDeploymentScript=../run_deployment.py
set PYTHONPATH=%projectPath%;%PYTHONPATH%
set config=deploy

python %runDeploymentScript% --config %config%

exit /b