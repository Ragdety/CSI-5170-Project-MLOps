@echo off


set stack_name=mlflow_stack
set tracker_name=mlflow_tracker
set deployer_name=mlflow

echo INFO: Registering stack with name %stack_name%...
zenml stack register %stack_name% -a default -o default -d %deployer_name% -e %tracker_name% --set

