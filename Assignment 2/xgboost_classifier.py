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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
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

random_state = np.random.RandomState(42)
outliers_fraction = 0.03
classifiers = {        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),}

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)     
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)

df_f=df[y_pred==0]
final = df_f.sample(frac=1).reset_index(drop=True)
y_final = final["target"].to_numpy()
X_final = final[["A","B","C","D","E","F","G"]].to_numpy()


under = RandomUnderSampler(sampling_strategy=1, random_state = 123)
X_res, y_res = under.fit_resample(X_final, y_final)


param_test1 = {'n_estimators':range(300,1000,100)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_res, y_res)

param_test2 = {'max_depth':range(3,12,2), 'min_child_weight':range(1,6,2)}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=900, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_res, y_res)

param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=900, max_depth=7, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_res, y_res)

param_test4 = {'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=900, max_depth=7, min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_res, y_res)

clf = XGBClassifier(learning_rate =0.1, n_estimators=900, max_depth=7, min_child_weight=1, gamma=0.1, subsample=0.9, colsample_bytree=0.9, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
clf.fit(X_res, y_res)



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
	with open("daksh_xgb.csv",'w',newline='') as csvfile:
    	csvwriter=csv.writer(csvfile)
    	csvwriter.writerow(fields)
    	csvwriter.writerows(rows)    

f = open("XGBClassifier.pkl", "wb")
pickle.dump(clf ,f)
f.close()






