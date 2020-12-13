"""
#導入資料
"""

import pandas as pd
import numpy as np
data=pd.read_csv('2019.csv')

data.columns

train=data[data['日期                  '].between('2019/10/01','2019/12/01')]

train=train.replace('NR',0)

test=data[5923:]
test=test.replace('NR',0)

train.isnull().sum()

train.isna().sum()

train.empty
#多個檢查發現沒有空值

len(train)

test.empty

"""
#填缺值

---
缺失值以及無效值以前後一小時平均值取代 (如果前一小時仍有空值，再取更前一小時)

NR表示無降雨，以0取代

"""

for i in range(len(train)):
    for a in range(3,len(train.iloc[i])):
        j = a-1
        k = a+1    
        if train.iloc[i,a].find('*') != -1 or train.iloc[i,a].find('#') != -1 or train.iloc[i,a].find('A') !=-1 or 
        train.iloc[i,a].find('x') !=-1:
            while train.iloc[j,a].find('*') != -1 or train.iloc[j,a].find('#') != -1 or train.iloc[j,a].find('A') !=-1 or 
            train.iloc[j,a].find('x') !=-1:
                j = j-1
            while str(train.iloc[k,a]).find('*') != -1 or str(train.iloc[k,a]).find('#') != -1 or str(train.iloc[k,a]).find('A') !=-1 or 
            train.iloc[k,a].find('x') !=-1:
                k = k+1
            train.iloc[i,a]=(eval(train.iloc[j,a])+eval(train.iloc[k,a]))/2
            #使用eval()字串翻譯大師

for i in range(len(test)):
    for a in range(3,len(test.iloc[i])):
        j = a-1
        k = a+1    
        if test.iloc[i,a].find('*') != -1 or test.iloc[i,a].find('#') != -1 or test.iloc[i,a].find('A') !=-1 or test.iloc[i,a].find('x') !=-1:
            while test.iloc[j,a].find('*') != -1 or test.iloc[j,a].find('#') != -1 or test.iloc[j,a].find('A') !=-1 or test.iloc[j,a].find('x') !=-1:
                j = j-1
            while str(test.iloc[k,a]).find('*') != -1 or str(test.iloc[k,a]).find('#') != -1 or str(test.iloc[k,a]).find('A') !=-1 or test.iloc[k,a].find('x') !=-1:
                k = k+1
            test.iloc[i,a]=(eval(test.iloc[j,a])+eval(test.iloc[k,a]))/2
            #使用eval()字串翻譯大師

"""
#切61天往右邊橫向延展

> TRAIN


"""

list_=train.iloc[:18,2:]
list_= list_.set_index(list_.iloc[:18,0], drop=True)
for i in range(2,62):
    list_2=train.iloc[18*(i-1):18*i,2:27]
    list_2= list_2.set_index(list_2.iloc[:18,0], drop=True)
    list_=pd.concat((list_,list_2),join='inner',axis=1)

list_=list_.drop(columns='測項                  ') #原使用inner做concat每天的資料開始左邊都有'測項'的欄位

array_=list_.values #dataframe轉成array的function

array_[17]
#最後一項測物10/1~11月底

"""
#切61天往右邊橫向延展

> TEST
"""

list_t=test.iloc[:18,2:]
list_t= list_t.set_index(list_t.iloc[:18,0], drop=True)
for i in range(2,32):
    list_2t=test.iloc[18*(i-1):18*i,2:27]
    list_2t= list_2t.set_index(list_2t.iloc[:18,0], drop=True)
    list_t=pd.concat((list_t,list_2t),join='inner',axis=1)

list_t=list_t.drop(columns='測項                  ') #原使用inner做concat每天的資料開始左邊都有'測項'的欄位

list_t.shape

array_t=list_t.values #dataframe轉成array的function

array_t[17]
#12月最後一項測物所有的值

"""
#分x與y

a.預測目標

     1. 將未來第一個小時當預測目標

         取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第6小時(未來第一小時)的PM2.5值(Y[0])，
         下一筆資料為第1~6小時的資料(X[1])去預測第7 小時的PM2.5值(Y[1])  *hint: 切割後X的長度應為1464-6=1458

     2. 將未來第六個小時當預測目標

         取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，去預測第11小時(未來第六小時)的PM2.5值(Y[0])，
         下一筆資料為第1~6小時的資料(X[1])去預測第12 小時的PM2.5值(Y[1])  *hint: 切割後X的長度應為1464-11=1453

 b. X請分別取

     1. 只有PM2.5 (e.g. X[0]會有6個特徵，即第0~5小時的PM2.5數值)

     2. 所有18種屬性 (e.g. X[0]會有18*6個特徵，即第0~5小時的所有18種屬性數值)

 c. 使用兩種模型 Linear Regression 和 Random Forest Regression 建模

 d. 用測試集資料計算MAE (會有8個結果， 2種X資料 * 2種Y資料 * 2種模型)

#X:18種屬性[0-5]預測未來第一小時[6]
"""

PM=array_.astype('float64') #這個可以將欄位轉成float
x0_train=[]
for i in range(1458): #最外圍[]
    box=[]
    k=i+6
    for j in range(18): #18個測物
        box.append([]) 
        for p in range(i,k): #依次共取6筆
            box[j].append(PM[j,p])
    x0_train.append(box)

len(x0_train) #檢查下長度

x0_train = np.array(x0_train).reshape((1458,18*6))

x0_train.shape

#基本上y只有兩種(6小跟11小的,就不重覆計算了)
y0_train=[]
for i in range(6,1464):
    y0_train.append(float(list_.iloc[9,i]))

"""#X:僅PM2.5[0-5]預測未來第一小時[6]"""

PM=list_.iloc[9]
x1_train=[]
y1_train=[]
for i in range(len(PM)-6):
    box=[]
    for j in range(6):
        box.append(float(PM[j]))
    x1_train.append(box)
    y1_train.append(float(PM[6]))
    PM=PM[1:]

y1_train==y0_train #基本上y只有兩種(6小跟11小的,就不重覆計算了)

"""
#X:18種屬性[0-5]預測未來第一小時[6]
"""

PM=array_t.astype('float64') #這個可以將欄位轉成float
x0_test=[]
for i in range(738): #最外圍[]
    box=[]
    k=i+6
    for j in range(18): #18個測物
        box.append([]) 
        for p in range(i,k): #依次共取6筆
            box[j].append(PM[j,p])
    x0_test.append(box) 
x0_test = np.array(x0_test).reshape((738,18*6))

x0_test.shape

"""
#X:僅PM2.5[0-5]預測未來第一小時[6]
"""

PM=list_t.iloc[9]
x1_test=[]
y1_test=[]
for i in range(len(PM)-6):
    box=[]
    for j in range(6):
        box.append(float(PM[j]))
    x1_test.append(box)
    y1_test.append(float(PM[6]))
    PM=PM[1:]

"""
#X:18種屬性[0-5]預測未來第六小時[11]
"""

PM=array_.astype('float64') #這個可以將欄位轉成float
x2_train=[]
for i in range(1453): #最外圍[]
    box=[]
    k=i+6
    for j in range(18): #18個測物
        box.append([]) 
        for p in range(i,k): #依次共取6筆
            box[j].append(PM[j,p])
    x2_train.append(box) 
x2_train = np.array(x2_train).reshape((1453,18*6))

"""
#X:僅PM2.5[0-5]未來第六小時[11]
"""

PM=list_.iloc[9]
x3_train=[]
y3_train=[]
for i in range(len(PM)-11):
    box=[]
    for j in range(6):
        box.append(float(PM[j]))
    x3_train.append(box)
    y3_train.append(float(PM[6]))
    PM=PM[1:]

"""
#X:18種屬性[0-5]預測未來第六小時[11]
"""

PM=array_t.astype('float64') #這個可以將欄位轉成float
x2_test=[]
for i in range(733): #最外圍[]
    box=[]
    k=i+6
    for j in range(18): #18個測物
        box.append([]) 
        for p in range(i,k): #依次共取6筆
            box[j].append(PM[j,p])
    x2_test.append(box) 
x2_test = np.array(x2_test).reshape((733,18*6))

"""
#X:僅PM2.5[0-5]未來第六小時[11]
"""

PM=list_t.iloc[9]
x3_test=[]
y3_test=[]
for i in range(len(PM)-11):
    box=[]
    for j in range(6):
        box.append(float(PM[j]))
    x3_test.append(box)
    y3_test.append(float(PM[6]))
    PM=PM[1:]

"""
#linear regression
"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x0_train, y1_train)
predictions = reg.predict(x0_test)
errors = abs(predictions - y1_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

reg = LinearRegression().fit(x1_train, y1_train)
predictions = reg.predict(x1_test)
errors = abs(predictions - y1_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

reg = LinearRegression().fit(x2_train, y3_train)
predictions = reg.predict(x2_test)
errors = abs(predictions - y3_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

reg = LinearRegression().fit(x3_train, y3_train)
predictions = reg.predict(x3_test)
errors = abs(predictions - y3_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

"""
#random forest
"""

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf.fit(x0_train, y1_train)
predictions = rf.predict(x0_test)
errors = abs(predictions - y1_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

rf.fit(x1_train, y1_train)
predictions = rf.predict(x1_test)
errors = abs(predictions - y1_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

rf.fit(x2_train, y3_train)
predictions = rf.predict(x2_test)
errors = abs(predictions - y3_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

rf.fit(x3_train, y3_train)
predictions = rf.predict(x3_test)
errors = abs(predictions - y3_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

