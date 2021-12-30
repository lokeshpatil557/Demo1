import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
path="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, names = headernames)
print("Null Value:",dataset.isnull().sum())
print(dataset)
print(dataset.info())
x =dataset.drop('Class',axis='columns')
y =dataset.Class
import sklearn.metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
print('train of x_train is:',X_train.shape)
print('train of y_train is:',y_train.shape)
print('train of x_test is:',X_test.shape)
print('train of y_test is:',y_test.shape)
from sklearn.ensemble import BaggingClassifier        #Baggingclassifier also known as Random Forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dtc = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=dtc,n_estimators= 100 ,random_state= 42)
results = cross_val_score(model,x,y,cv= 10)
print(results.mean())

# AdaBoost Classification
 
from sklearn.ensemble import AdaBoostClassifier
 
model = AdaBoostClassifier(base_estimator=dtc,n_estimators=100, random_state= 42)
results = cross_val_score(model,x,y,cv= 10)
print(results.mean())

#Stacking
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

# create the sub models
estimators = []
model1 = GaussianNB()
estimators.append(('Naive_Bais', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble,x,y, cv= 10)
print(results.mean())
print(ensemble)

#GradientBoostingClassifier


# importing machine learning models for prediction
from sklearn.ensemble import GradientBoostingClassifier
# initializing the boosting module with default parameters
model = GradientBoostingClassifier()
#model = AdaBoostClassifier(n_estimators=100, random_state= 42)
results = cross_val_score(model,x,y, cv= 10)
print(results.mean())





















