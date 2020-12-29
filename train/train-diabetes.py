from azureml.core import Workspace, Dataset
from azureml.core.run import Run

from interpret.glassbox import ExplainableBoostingRegressor
from interpret import preserve

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import os
import joblib

# メトリック記録開始
run = Run.get_context()

# Workspace のオブジェクト
ws = run.experiment.workspace

# データの準備 (クラウドのストレージのマウント、データダウンロードも可能)
df = Dataset.get_by_name(ws, 'diabetesData').to_pandas_dataframe()
X = df.drop(['Y'], axis=1)
y = df['Y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ハイパーパラメータ
interactions = 10
run.log("interactions", interactions)

# モデル学習
ebm = ExplainableBoostingRegressor(interactions=interactions)
ebm.fit(X_train, y_train)

# モデル精度
preds = ebm.predict(X_test)
mse = mean_squared_error(y_test, preds)
print("mse:", mse)
run.log("mse", mse)

# 結果保存
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/html', exist_ok=True)


ebm_global = ebm.explain_global(name='EBM')

preserve(ebm_global, file_name="outputs/html/global-importance.html")
for i in ebm_global.selector.Name:
    #print(i)
    preserve(ebm_global, i, file_name="outputs/html/"+i+".html")


# モデルファイルの保存
model_name = "diabetes-model.pkl"
joblib.dump(value=ebm, filename='outputs/' + model_name)