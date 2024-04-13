@echo off

set projectPath=../../../
set runPipeScript=../run_pipeline.py
set PYTHONPATH=%projectPath%;%PYTHONPATH%

python %runPipeScript%

exit /b