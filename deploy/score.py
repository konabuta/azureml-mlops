import numpy as np
import json
import os
import joblib
from interpret.glassbox import ExplainableBoostingRegressor
from azureml.core.model import Model


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'diabetes-model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    y_hat = model.predict(data)
    return y_hat.tolist()