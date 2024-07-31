import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits

data = load_digits()
print(dir(data))
print("scores of logisitc regression model:",cross_val_score(LogisticRegression(max_iter=3000),data.data,data.target,cv=3))
print("scores of svc model:",cross_val_score(SVC(C=10),data.data,data.target,cv=3))
print("scores of random forest classifier: ",cross_val_score(RandomForestClassifier(n_estimators=40),data.data,data.target,cv=3))