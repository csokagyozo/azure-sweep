# azure-sweep
A test hyperparameter search in Azure

## prerequisites:
0. logged in to azure
1. create workspace and resource group in Azure Machine Learning Studio
2. create a compute instance
3. create env: tensorflow from docker image tensorflow/tensorflow

## run from CLI
4. ``python azure_sweep.py --subscription_id <SUBSCRIPTIONI_ID> --resource_group <RG> --workspace <WS>``
5. (after finished) ``az ml job download --name <JOB_NAME> --all --resource-group <RG> --workspace-name <WS>``  

