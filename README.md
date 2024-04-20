# CSI-5170-Project-MLOps
The purpose of this project is to learn about MLOps pipelines applied to basic machine learning algorithms. 

This is the final project for class "CSI-5170 - Machine Learning" at Oakland University. We decided to take our industry experience and apply it into a more practical project to enhance our skills in Machine Learning Operations, the future of ML!


# Setup
To get started with this project, execute:
* [`setup.bat`](setup.bat) installs the main dependencies and configures them.

The main dependencies include:
* ZenML (https://www.zenml.io) - MLOps framework used in this project.

After main dependencies are setup, execute:
* [`install_py_deps.bat`](install_py_deps.bat) installs all python dependencies used within the entire project. 

>**Note**: [`install_py_deps.bat`](install_py_deps.bat) can be re-run or used as many times as needed, it will re-generate the requirements.txt automatically and install python dependencies.

After python dependencies are installed, open separate terminal and execute:
* [`up-zenml.bat`](mlops/zenml/scripts/up-zenml.bat) initializes and runs zenML server in localhost. 


# Executing MLOps Pipeline
To run the MLOps pipeline, open separate terminal and execute:
* [`run_pipeline.bat`](mlops/zenml/scripts/run_pipeline.bat) sets up python path to the root of this project for modules to be imported correctly and runs the mlops pipeline.
