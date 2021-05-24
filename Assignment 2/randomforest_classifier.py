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
from sklearn.model_selection import GridSearchCV
from collections import Counter
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("given_dataset.csv")
df = df.drop(columns = ["id"])
df = df.rename(columns={"T": "target"})

# print(df.target)

grouped = df.groupby(df.target)
zeroes = grouped.get_group(0)
ones = grouped.get_group(1)
# print(zeroes)
# print(ones)

y = df["target"].to_numpy()
X = df[["A","B","C","D","E","F","G"]].to_numpy()
# print(y)
# print(X)


min_max_scaler = MinMaxScaler(feature_range = (0,10))
X = min_max_scaler.fit_transform(X)
# print(X)

random_state = np.random.RandomState(42)
outliers_fraction = 0.03
classifiers = {'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),}

for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)     
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    # print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)


df_f=df[y_pred==0]
final = df_f.sample(frac=1).reset_index(drop=True)
y_final = final["target"].to_numpy()
X_final = final[["A","B","C","D","E","F","G"]].to_numpy()    




undersample = RandomUnderSampler(sampling_strategy='majority')
X_over, y_over = undersample.fit_resample(X_final, y_final)


param_test1 = {'n_estimators':range(100,700,100)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_depth=20, n_estimators = 200, bootstrap=True, random_state=0, max_features = 4, min_samples_split = 8,min_samples_leaf = 2), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_over, y_over)

param_test2 = {'max_depth':range(8,26,5), 'min_samples_split':range(1,100,10)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(max_depth=20, n_estimators = 700, bootstrap=True, random_state=0, max_features = 4, min_samples_split = 8,min_samples_leaf = 2), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_over, y_over)

param_test3 = {'min_samples_leaf':range(1,3,15)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(max_depth=23, n_estimators = 700, bootstrap=True, random_state=0, max_features = 4, min_samples_split = 11,min_samples_leaf = 2), 
param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_over, y_over)

param_test4 = {'max_features':[3,4,5,6,7,8]}
gsearch4 = GridSearchCV(estimator =  RandomForestClassifier(max_depth=23, n_estimators = 700, bootstrap=True, random_state=0, max_features = 4, min_samples_split = 11,min_samples_leaf = 1), 
param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_over, y_over)


clf = RandomForestClassifier(max_depth=23, n_estimators = 700, bootstrap=True, random_state=0, max_features = 4, min_samples_split = 11,min_samples_leaf = 1)
clf.fit(X_over, y_over)


def gplot(trainhist, testhist):
    ind = [i for i in range(1,8)]
    plt.plot(ind , trainhist, label = 'training accuracy') #Plot for cost of training samples
    plt.plot(ind , testhist, label = 'testing accuracy') #Plot for cost of validation samples
    plt.xlabel("Max Features") #X axis label
    plt.ylabel("Accuracy") #Yaxis label
    plt.legend() #To show the legend
    plt.savefig("max_features_rf", bbox_inches='tight')
    plt.show() #Show the plot
# Runner Function
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
	with open("daksh_rf_classifier.csv",'w',newline='') as csvfile:
    	csvwriter=csv.writer(csvfile)
    	csvwriter.writerow(fields)
    	csvwriter.writerows(rows)    



f = open("RandomForestClassifier.pkl", "wb")
pickle.dump(clf ,f)
f.close()



X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, stratify = y_over, random_state = 123, test_size = 0.2)
l_train = []
l_test = []
for i in range(1, 8):
    clf = RandomForestClassifier(max_depth=23, n_estimators = 700, bootstrap=True, random_state=0, max_features = i, min_samples_split = 11,min_samples_leaf = 1)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    l_train.append(accuracy_score(y_train, y_pred_train))
    l_test.append(accuracy_score(y_test, y_pred_test))

gplot(l_train, l_test)