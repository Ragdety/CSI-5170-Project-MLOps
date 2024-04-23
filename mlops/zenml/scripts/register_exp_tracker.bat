@echo off


set tracker_name=mlflow_tracker

echo INFO: Registering experiment tracker with name %tracker_name%...
zenml experiment-tracker register %tracker_name% --flavor=mlflow

