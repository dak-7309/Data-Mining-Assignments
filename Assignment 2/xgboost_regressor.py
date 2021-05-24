import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import csv
import imblearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import pickle

df = pd.read_csv("given_dataset.csv")
df = df.drop(columns = ["id"])
df = df.rename(columns={"T": "target"})

grouped = df.groupby(df.target)
zeroes = grouped.get_group(0)
ones = grouped.get_group(1)
y = df["target"].to_numpy()
X = df[["A","B","C","D","E","F","G"]].to_numpy()

min_max_scaler = MinMaxScaler(feature_range = (0,10))
X = min_max_scaler.fit_transform(X)

final = df.sample(frac=1).reset_index(drop=True)
y_final = final["target"].to_numpy()
X_final = final[["A","B","C","D","E","F","G"]].to_numpy()



undersample = RandomUnderSampler(sampling_strategy=0.5)
X_over, y_over = undersample.fit_resample(X_final, y_final)




clf = XGBRegressor(learning_rate =0.1, n_estimators=900, max_depth=7, min_child_weight=1, gamma=0.1, subsample=0.9, colsample_bytree=0.9, objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27)
clf.fit(X_over, y_over)


# define runner function
def runner(clf):
	predict_this = pd.read_csv("to_predict.csv")
	idss = predict_this["id"].to_numpy()
	predict_this.pop("id")

	res = clf.predict(predict_this.to_numpy())
	res = np.reshape(res, (-1, 1))
	idss = np.reshape(idss, (-1, 1))

	fields=["id","T"]
	rows=[]
	for i in range(len(idss)):
    	rows.append([idss[i][0],res[i][0]])
	with open("daksh_xgb_regressor.csv",'w',newline='') as csvfile:
    	csvwriter=csv.writer(csvfile)
    	csvwriter.writerow(fields)
    	csvwriter.writerows(rows)    

f = open("XGBRegressor.pkl", "wb")
pickle.dump(clf ,f)
f.close()
