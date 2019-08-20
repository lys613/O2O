#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


# In[11]:


dataset1 = pd.read_csv('dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('dataset3.csv')


# In[12]:


print(dataset3.shape,dataset2.shape,dataset2.shape)


# In[13]:


pd.set_option('display.max_columns',200)   
dataset1.head()


# In[14]:


dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)


# In[15]:


#把前面两个数据集整合在一起
dataset12 = pd.concat([dataset1,dataset2],axis=0)


# In[16]:


dataset1_y = dataset1.label
dataset1_x = dataset1.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)  
# 'day_gap_before','day_gap_after' cause overfitting, 0.77


# In[17]:


dataset2_y = dataset2.label
dataset2_x = dataset2.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)


# In[18]:


dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)


# In[90]:


dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)
dataset3_preds.shape
dataset3_x.shape


# In[111]:


#将空值用0填充
dataset1_x=dataset1_x.fillna(0)
dataset2_x=dataset2_x.fillna(0)
dataset12_x=dataset12_x.fillna(0)
dataset3_x=dataset3_x.fillna(0)
dataset3_x.shape


# In[ ]:


##官方的用法


# In[98]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


# In[118]:


params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


# In[104]:


dataset12_y.shape


# In[119]:


clf.fit(dataset1_x, dataset1_y) #训练
mse = mean_squared_error(dataset2_y, clf.predict(dataset2_x)) #预测并且计算MSE
mse


# In[141]:


preds= clf.predict(dataset3_x)
preds.reshape(-1,1)
preds= pd.DataFrame(preds)
preds.head()


# In[148]:


preds.to_csv('D:/天池/GBDT_label.csv',index=None)


# In[149]:


preds.shape


# In[ ]:





# In[70]:


trainSub=dataset12_x
validSub=dataset12_y


# In[71]:


#trainSub, validSub = train_test_split(dataset12_x, test_size = 0.2, stratify = dataset12_y, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(dataset12_x, dataset12_y) #默认是75%训练集
x_train.shape
x_test.shape


# In[79]:


# 模型训练，使用GBDT算法
gbr = GradientBoostingClassifier(
                        n_estimators=1000, 
                        max_depth=5, 
                        min_samples_split=2, 
                        learning_rate=0.01)
gbr.fit(x_train, y_train.ravel())
#joblib.dump(gbr, 'train_model_result4.m')   # 保存模型


# In[80]:


#训练和验证
y_gbr = gbr.predict(x_train)
y_gbr1 = gbr.predict(x_test)
acc_train = gbr.score(x_train, y_train)
acc_test = gbr.score(x_test, y_test)
print(acc_train)
print(acc_test)
y_gbr1.shape


# In[87]:


preds= gbr.predict(dataset3_x)
preds=preds.reshape(-1,1)
preds.shape


# In[88]:


preds = pd.DataFrame(preds)
#preds.rename(columns={'0':'label'}, inplace = True)
preds.shape
preds.head()


# In[89]:


preds.to_csv("gbdt_label.csv",index=None,header=None)


# In[91]:


dataset3_preds.to_csv("gbdt_dataset3.csv",index=None,header=None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




