

"""
#前處理
"""

import pandas as pd
import numpy as np
hw2=pd.read_csv('HW2data.csv')
# hw2.replace(' ?', np.nan, inplace=True)

hw2.workclass.value_counts()

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
hw2['workclass'] = class_le.fit_transform(hw2['workclass'].values)
hw2.workclass.value_counts()

hw2['marital_status']=class_le.fit_transform(hw2['marital_status'].values)
hw2['occupation']=class_le.fit_transform(hw2['occupation'].values)
hw2['relationship']=class_le.fit_transform(hw2['relationship'].values)
hw2['race']=class_le.fit_transform(hw2['race'].values)
hw2['sex']=class_le.fit_transform(hw2['sex'].values)
hw2['native_country']=class_le.fit_transform(hw2['native_country'].values)
hw2['income']=class_le.fit_transform(hw2['income'].values)

hw2.income.value_counts()
#<=50K為0類 ,>50K為1類

"""
#kfold
"""

#切kfold,除了第i個fold作為testfold外,其餘合併成為trainfold
def K_fold_CV(k,i,data):
    global test_fold,train_fold
    fold_len=int(len(data)/k) #加了一個int可以避免資料筆數沒除盡的情況
    fold_start=int(i*fold_len)
    fold_end=int((i+1)*fold_len)-1
    test_fold=data.loc[fold_start:fold_end]  
    if fold_start==0:
        train_fold=data.loc[fold_end+1:]
    else:
        if fold_start==(len(data)-fold_len):
            train_fold=data.loc[:fold_start-1]
        else:
            train_fold=pd.concat([data.loc[:fold_start-1],data.loc[fold_end+1:]],axis=0)

# X=iris.drop(['Id','Species'],axis=1)
# y=iris['Species']
x=hw2.drop(['income','education'],axis=1)
y=hw2['income']

"""
#+random forest, avg accuracy
"""

from pandas import DataFrame
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
rfc=RandomForestClassifier()

k=10
Accuracy=[]
for i in range(k):
    K_fold_CV(k,i,x)
    X_test=test_fold
    X_train=train_fold
    K_fold_CV(k,i,y)
    y_test=test_fold
    y_train=train_fold
    rfc.fit(X_train,y_train)
    y_hat = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)
    print(accuracy)
    Accuracy.append(accuracy)
    i+=1    
Avg_accuarcy=np.mean(Accuracy)
print('10次的平均accuarcy:')
print(Avg_accuarcy)

#用商 for i in range(10):
#n=len(df)//k
#x=data[n:]
#y=data[]
