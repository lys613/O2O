#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


# In[20]:


dataset1 = pd.read_csv('dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('dataset3.csv')


# In[21]:


print(dataset3.shape,dataset2.shape,dataset2.shape)


# In[22]:


pd.set_option('display.max_columns',200)   
dataset1.head()


# In[23]:


dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)


# In[24]:


#把前面两个数据集整合在一起
dataset12 = pd.concat([dataset1,dataset2],axis=0)


# In[25]:


dataset1_y = dataset1.label
dataset1_x = dataset1.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)  
# 'day_gap_before','day_gap_after' cause overfitting, 0.77


# In[26]:


dataset2_y = dataset2.label
dataset2_x = dataset2.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)


# In[27]:


dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)


# In[28]:


dataset3_preds = dataset3[['user_id','coupon_id','date_received']]
dataset3_x = dataset3.drop(['user_id','coupon_id','date_received','day_gap_before','day_gap_after'],axis=1)


# In[29]:


print(dataset1_x.shape,dataset2_x.shape,dataset3_x.shape)


# In[30]:


dataset1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
dataset2 = xgb.DMatrix(dataset2_x,label=dataset2_y)
dataset12 = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3 = xgb.DMatrix(dataset3_x)


# In[31]:


params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }


# In[32]:


watchlist = [(dataset12,'train')]
model = xgb.train(params,dataset12,num_boost_round=3500,evals=watchlist)


# In[33]:


dataset3_preds['label'] = model.predict(dataset3)
dataset3_preds.label=np.array(dataset3_preds.label).reshape(-1,1)
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
dataset3_preds.label=dataset3_preds[['label']].apply(max_min_scaler)
dataset3_preds


# In[34]:


dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
dataset3_preds.head()


# In[64]:


#predict test set
dataset3_preds['label'] = model.predict(dataset3)
#dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.reshape(-1, 1))
#dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label)

dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
#print(dataset3_preds.describe())
dataset3_preds.head()


# In[65]:


#save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)


# In[ ]:





# In[ ]:




