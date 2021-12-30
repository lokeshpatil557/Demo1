import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
a=pd.read_csv("C:/Users/Lokesh/Desktop/cdac ai material/machine learning/lab/creditcard.csv")
print(a.head())
a.drop(('Time'),axis=1,inplace=True)
a.drop(('Amount'),axis=1,inplace=True)
print(a.info())
x=a.drop('Class',axis=1)
y=a['Class']
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=100)
print('train of x_train is:',x_train.shape)
print('train of y_train is:',y_train.shape)
print('train of x_test is:',x_test.shape)
print('train of y_test is:',y_test.shape)
model = DecisionTreeClassifier()
print(model)
clf = model.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of model is:",accuracy)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
print(model)
dlf = model.fit(x_train,y_train)
y_predicted = dlf.predict(x_test)
print(y_predicted)
print(y_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predicted)
print("Accuracy of model is:",accuracy)
from sklearn.metrics import classification_report
cr=classification_report(y_test,y_predicted)
print(cr)



list1=[20,30,40,50]
list2=['gini','entropy']
from sklearn.model_selection import GridSearchCV
para_grid={'n_estimators' : list1,'criterion':list2}
gridsearch=GridSearchCV(model,para_grid,cv=5)
gridsearch.fit(x_train,y_train)
print(gridsearch.best_params_)
print(gridsearch.best_score_)
print('Accuracy:',metrics.accuracy_score(y_test,y_predicted))
















