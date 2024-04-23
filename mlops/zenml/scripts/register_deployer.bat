@echo off


set deployer_name=mlflow

echo INFO: Registering model deployer with name %deployer_name%...
zenml model-deployer register %deployer_name% --flavor=mlflow



