# ''''''
# Microsoft MLOps サンプルコードを参考
# https://github.com/microsoft/MLOps/blob/master/examples/cli-train-deploy/generate-runconfig.py
# ''''''

from azureml.core import RunConfiguration, ScriptRunConfig, Dataset, Workspace, Environment
from azureml.core.runconfig import Data, DataLocation, Dataset as RunDataset
from azureml.core.script_run_config import get_run_config_from_script_run
from azureml.core.conda_dependencies import CondaDependencies

# Connect to the workspace
ws = Workspace.from_config()

# Create a new environment and set Conda dependencies
conda_env = Environment.get(ws, 'diabetes-env')
# conda_env.python.conda_dependencies = CondaDependencies.create(
#     pin_sdk_version=False,
#     pip_packages=[
#         'scikit-learn',
#         'azureml-sdk',
#         'azureml-dataprep[pandas,fuse]'
#     ])

# Get the dataset that will be used
dataset = Dataset.get_by_name(ws, 'diabetesData')
# Define the environment variable/where data will be mounted
input_name = 'diabetesData'
# Define the name of the compute target for training
compute_name = 'cpuclusters'

# Define the script run config
src = ScriptRunConfig(
    source_directory='train',
    script='train-diabetes.py',
#     arguments=[
#         '--data-folder',
#         'DatasetConsumptionConfig:{}'.format(input_name)
#     ]
)

# Define the data section of the runconfig
# src.run_config.data = {
#     input_name: Data(
#         data_location=DataLocation(
#             dataset=RunDataset(dataset_id=dataset.id)),
#         create_output_directories=False,
#         mechanism='mount',
#         environment_variable_name=input_name,
#         overwrite=False
#     )
# }
# Set other parameters for the run
src.run_config.framework = 'python'
src.run_config.environment = conda_env
src.run_config.target = compute_name

# Save the run configuration as a .azureml/mnist.runconfig
get_run_config_from_script_run(src).save(name='diabetes.runconfig')