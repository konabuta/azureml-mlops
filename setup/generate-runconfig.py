import os
from azureml.core import RunConfiguration, Workspace, Environment

ws = Workspace.from_config()
conda_env = Environment.get(ws, 'diabetes-env')
compute_name = 'cpuclusters'

run_config = RunConfiguration()

run_config.framework = 'python'
run_config.environment = conda_env
run_config.target = compute_name

#リポジトリ直下から実行
os.makedirs("./.azureml", exist_ok=True)
run_config.save(name='diabetes.runconfig', path="./")
