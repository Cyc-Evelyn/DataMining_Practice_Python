"""#Using XGBoost,GBDT,LightGBM
---
1.   split into train/test dataset
2.   use tfidf in sklearn
3. reindex testY into 0-1247 make testX more easy to recognize
"""

from sklearn.feature_extraction.text import TfidfVectorizer 
import gensim
#tfidf
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=2500, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(clean_data['clean_text'])
tfidf.shape

trainY = clean_data['label'][:4987]
testY = clean_data['label'][4987:]

#traonsfrom train label datatype & reindex test index
trainY = trainY.astype(int)
testY = testY.reset_index()
testY = testY.reset_index(drop=True)
testY=testY.drop(columns=['index'])
testY=pd.Series(testY['label'].values)

#80% as train,20% as test(like original setting)
trainX_data = tfidf[:4987]
testX_data = tfidf[4987:]
#because x in tfidf only got len but no index, so we collect it from y
xtrain_tfidf = trainX_data[trainY.index]
xvalid_tfidf = testX_data[testY.index]

from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression(solver='lbfgs') 
lreg.fit(xtrain_tfidf, trainY)
prediction = lreg.predict_proba(xvalid_tfidf)

prediction = lreg.predict_proba(xvalid_tfidf)

prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
#XGBoost
#heigher learning rate perform better
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
xgb = XGBClassifier(max_depth=6, n_estimators=500,learning_rate=0.3).fit(xtrain_tfidf, trainY) 
prediction = xgb.predict(xvalid_tfidf)
testY=testY.astype(np.int)
f1_score(testY, prediction)

#Other method seting threshold:  
# >0.5 =>1
# y_pred = (y_pred >= 0.5)*1 
# y_pred

from sklearn.metrics import classification_report
print('XGBoost : ')
print(classification_report(testY, prediction))

#GBDT
#heigher learning rate perform better
from sklearn.ensemble import GradientBoostingClassifier  
clf= GradientBoostingClassifier(learning_rate=1,max_depth=1, random_state=0).fit(xtrain_tfidf, trainY)  
prediction_GBDT = clf.predict(xvalid_tfidf)
testY=testY.astype(np.int)
f1_score(testY, prediction_GBDT)

print('GBDT : ')
print(classification_report(testY, prediction_GBDT))

#LightGBM
#lower learning rate perform better
#heigher estimators perform worse
import lightgbm as lgb
lgb_clf = lgb.LGBMClassifier(learning_rate=0.2,num_leaves=35)
lgb_clf.fit(xtrain_tfidf, trainY)
prediction_lgbm =lgb_clf.predict(xvalid_tfidf)
testY=testY.astype(np.int)
f1_score(testY, prediction_lgbm)

print('LightGBM : ')
print(classification_report(testY, prediction_lgbm))

"""#小結
---
1.   也許可以取特定詞頻的字過濾?   
ex.本文件中總頻率為基準,高頻約5000-6000次,中頻2000-5000,低頻600-2000,稀有字可能僅出現在某文檔中(人名)
2.   去掉shortword可能會減低一部份的前後文/片語資訊
3.   依據幾份文件出現該詞的方式過濾,而非總詞頻
"""
