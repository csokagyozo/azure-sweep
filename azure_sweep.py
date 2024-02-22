from azure.ai.ml.sweep import Choice
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command, Input

import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription_id')
    parser.add_argument('--resource_group')
    parser.add_argument('--workspace')
    parser.add_argument('--compute')
    parser.add_argument('--job_name')
    args = parser.parse_args()
    return args.subscription_id, args.resource_group, args.workspace, args.compute, args.job_name


subscription_id, resource_group, workspace, compute, job_name = get_arguments()

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

command_job = command(
    code='./src',
    command='python mnist_azure_test.py --epochs 15 --learning_rate ${{inputs.learning_rate}}',
    environment='tensorflow@latest',
    inputs={
        'learning_rate': 0.9
    },
    compute=compute
)

command_job_for_sweep = command_job(
    learning_rate=Choice(values=[0.0001, 0.0003, 0.001, 0.003, 0.01])
)

sweep_job = command_job_for_sweep.sweep(
    compute=compute,
    sampling_algorithm = "grid",
    primary_metric="accuracy",
    goal="Maximize"
)

sweep_job.display_name = job_name
sweep_job.experiment_name = "mnist-sweep-example"
sweep_job.description = "Run a hyperparameter sweep job for Adam on MNIST dataset."
sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=3, timeout=12600)

returned_sweep_job = ml_client.create_or_update(sweep_job)
print(returned_sweep_job.services["Studio"].endpoint)

