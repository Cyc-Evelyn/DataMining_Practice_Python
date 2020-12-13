"""
#取資料
"""

import numpy as np
import pandas as pd
death=pd.read_csv('character-deaths.csv')
death.isnull().sum()

"""
#前處理 :選擇'Book of Death'為y後
drop 'Death Year','Death Chapter'
"""

death.drop(['Death Year','Death Chapter'],axis=1,inplace=True)
death['Book of Death'].fillna(0,inplace=True)
death['Book Intro Chapter'].fillna(0,inplace=True)

death.head()

death["Book of Death"].loc[death["Book of Death"]!=0]=1

death=death.join(pd.get_dummies(death.Allegiances))

death.shape #增加21個特徵(21姓氏)

death.head()

death=death.drop(columns='Allegiances')

"""#decision tree
"""

X=death.drop(['Book of Death','Name'],axis=1)
y=death['Book of Death']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
parameters={'min_samples_split' : range(10,100,10),'max_depth': range(1,20,2)}
clf_tree=DecisionTreeClassifier()
clf=GridSearchCV(clf_tree,parameters)
clf.fit(X_train,y_train)
clf.best_estimator_

classer=DecisionTreeClassifier(min_samples_split=70,max_depth=9)
classer.fit(X_train ,y_train)

y_pred = classer.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
#繪圖-graphviz
"""

import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus

dot_data = export_graphviz(classer, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('death.pdf')
