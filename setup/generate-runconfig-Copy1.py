# -*- coding: utf-8 -*-
# ''''''
# Microsoft MLOps サンプルコードを参考
# https://github.com/microsoft/MLOps/blob/master/examples/cli-train-deploy/generate-runconfig.py
# ''''''

from azureml.core import RunConfiguration, ScriptRunConfig, Dataset, Workspace, Environment
from azureml.core.runconfig import Data, DataLocation, Dataset as RunDataset
from azureml.core.script_run_config import get_run_config_from_script_run

ws = Workspace.from_config()
conda_env = Environment.get(ws, 'diabetes-env')
dataset = Dataset.get_by_name(ws, 'diabetesData')
input_name = 'diabetesData'
compute_name = 'cpuclusters'

src.run_config.framework = 'python'
src.run_config.environment = conda_env
src.run_config.target = compute_name

get_run_config_from_script_run(src).save(name='diabetes.runconfig')
