#!/usr/bin/env python
# coding: utf-8

# In[223]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None)

df = pd.read_csv('data.csv')


# In[224]:


df.median()


# In[225]:


df = df.drop(columns='ISO639-3 codes')
df = df.drop(columns='Name in French')
df = df.drop(columns='Name in Spanish')
df = df.drop(columns='Countries')
df = df.drop(columns='Alternate names')
df = df.drop(columns='Name in the language')
df = df.drop(columns='Sources')
df = df.drop(columns='Description of the location')
df.head()


# In[226]:


df[['country_0','country_1','country_2','country_3','country_4','country_5','country_6','country_7','country_8','country_9','country_10','country_11','country_12','country_13','country_14','country_15','country_16','country_17','country_18','country_19','country_20','country_21','country_22','country_23','country_24','country_25','country_26','country_27','country_28']]=df['Country codes alpha 3'].str.split(',', expand = True)


# In[227]:


df['country_1'] = df['country_1']. replace(np. nan, 0)
df['country_2'] = df['country_2']. replace(np. nan, 0)
df['country_3'] = df['country_3']. replace(np. nan, 0)
df['country_4'] = df['country_4']. replace(np. nan, 0)
df['country_5'] = df['country_5']. replace(np. nan, 0)
df['country_6'] = df['country_6']. replace(np. nan, 0)
df['country_7'] = df['country_7']. replace(np. nan, 0)
df['country_8'] = df['country_8']. replace(np. nan, 0)
df['country_9'] = df['country_9']. replace(np. nan, 0)
df['country_10'] = df['country_10']. replace(np. nan, 0)
df['country_11'] = df['country_11']. replace(np. nan, 0)
df['country_12'] = df['country_12']. replace(np. nan, 0)
df['country_13'] = df['country_13']. replace(np. nan, 0)
df['country_14'] = df['country_14']. replace(np. nan, 0)
df['country_15'] = df['country_15']. replace(np. nan, 0)
df['country_16'] = df['country_16']. replace(np. nan, 0)
df['country_17'] = df['country_17']. replace(np. nan, 0)
df['country_18'] = df['country_18']. replace(np. nan, 0)
df['country_19'] = df['country_19']. replace(np. nan, 0)
df['country_20'] = df['country_20']. replace(np. nan, 0)
df['country_21'] = df['country_21']. replace(np. nan, 0)
df['country_22'] = df['country_22']. replace(np. nan, 0)
df['country_23'] = df['country_23']. replace(np. nan, 0)
df['country_24'] = df['country_24']. replace(np. nan, 0)
df['country_25'] = df['country_25']. replace(np. nan, 0)
df['country_26'] = df['country_26']. replace(np. nan, 0)
df['country_27'] = df['country_27']. replace(np. nan, 0)
df['country_28'] = df['country_28']. replace(np. nan, 0)


# In[228]:


df['country_0'] = np.where(df['country_0'] != 0, 1, 0)
df['country_1'] = np.where(df['country_1'] != 0, 1, 0)
df['country_2'] = np.where(df['country_2'] != 0, 1, 0)
df['country_3'] = np.where(df['country_3'] != 0, 1, 0)
df['country_4'] = np.where(df['country_4'] != 0, 1, 0)
df['country_5'] = np.where(df['country_5'] != 0, 1, 0)
df['country_6'] = np.where(df['country_6'] != 0, 1, 0)
df['country_7'] = np.where(df['country_7'] != 0, 1, 0)
df['country_8'] = np.where(df['country_8'] != 0, 1, 0)
df['country_9'] = np.where(df['country_9'] != 0, 1, 0)
df['country_10'] = np.where(df['country_10'] != 0, 1, 0)
df['country_11'] = np.where(df['country_11'] != 0, 1, 0)
df['country_12'] = np.where(df['country_12'] != 0, 1, 0)
df['country_13'] = np.where(df['country_13'] != 0, 1, 0)
df['country_14'] = np.where(df['country_14'] != 0, 1, 0)
df['country_15'] = np.where(df['country_15'] != 0, 1, 0)
df['country_16'] = np.where(df['country_16'] != 0, 1, 0)
df['country_17'] = np.where(df['country_17'] != 0, 1, 0)
df['country_18'] = np.where(df['country_18'] != 0, 1, 0)
df['country_19'] = np.where(df['country_19'] != 0, 1, 0)
df['country_20'] = np.where(df['country_10'] != 0, 1, 0)
df['country_21'] = np.where(df['country_21'] != 0, 1, 0)
df['country_22'] = np.where(df['country_22'] != 0, 1, 0)
df['country_23'] = np.where(df['country_23'] != 0, 1, 0)
df['country_24'] = np.where(df['country_24'] != 0, 1, 0)
df['country_25'] = np.where(df['country_25'] != 0, 1, 0)
df['country_26'] = np.where(df['country_26'] != 0, 1, 0)
df['country_27'] = np.where(df['country_27'] != 0, 1, 0)
df['country_28'] = np.where(df['country_28'] != 0, 1, 0)


# In[229]:


df['Number of speakers'] = df['Number of speakers']. replace(np. nan, 800)
df['Latitude'] = df['Latitude']. replace(np. nan, 15.6587)
df['Longitude'] = df['Longitude']. replace(np. nan, 30.5914)


# In[230]:


num = df['Number of speakers']
name= df['Name in English']
fig = plt.figure(figsize =(10, 7))
plt.bar(name[0:10], num[0:10])
plt.show()


# In[231]:


df = df.drop(columns='Country codes alpha 3')
df = df.drop(columns='Name in English')
df = df.drop(columns='ID')


# In[232]:


train, dev, test = np.split(df.sample(frac = 1, random_state = 42), [int(.7*len(df)), int(.85*len(df))])


# In[233]:


X_train = (train.drop(columns = ["Degree of endangerment"])).to_numpy()
y_train = train["Degree of endangerment"].to_numpy()

X_dev = (dev.drop(columns = ["Degree of endangerment"])).to_numpy()
y_dev = dev["Degree of endangerment"].to_numpy()

X_test = (test.drop(columns = ["Degree of endangerment"])).to_numpy()
y_test = test["Degree of endangerment"].to_numpy()


# In[234]:


# feature scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_dev_std = sc.transform(X_dev)
X_test_std = sc.transform(X_test)


# In[246]:


svm = SVC()
svm.fit(X_dev_std, y_dev)
print("Accuracy is: ", svm.score(X_dev_std, y_dev))


# In[241]:


parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[0.00001, 0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_dev_std, y_dev)


# In[245]:


clf.best_params_


# In[247]:


clf.best_score_


# In[260]:


svc = SVC(C = 1000.0, kernel = 'rbf')
svc.fit(X_train_std, y_train)


# In[261]:


y_pred = svm.predict(X_test_std)


# In[262]:


print(classification_report(y_test, y_pred))


# In[ ]:




