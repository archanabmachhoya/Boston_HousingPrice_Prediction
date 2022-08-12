#!/usr/bin/env python
# coding: utf-8

# In[44]:


###importing Libraries


# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


#Importing the Boston Housing dataset


# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[3]:


data = pd.DataFrame(boston.data)


# In[4]:


data.head()


# In[46]:


#Adding the feature names to the dataframe


# In[5]:


data.columns = boston.feature_names
data.head()


# In[ ]:


#Adding target variable to dataframe


# In[6]:


data['PRICE'] = boston.target


# In[7]:


data.shape


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# In[ ]:


#Plotting the heatmap of correlation between features


# In[14]:


corr = data.corr()
f, ax = plt.subplots(figsize=(18, 15))
fig_corr = sns.heatmap(corr, annot=True)
plt.title("Correlation")
plt.show()


# In[16]:


## Splitting to training and testing data


# In[18]:


X = data.loc[:, data.columns != 'PRICE']
y = data.loc[:, data.columns == 'PRICE']


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)


# In[ ]:


## Apply Model


# In[20]:


from sklearn.linear_model import LinearRegression


# In[25]:


lr = LinearRegression()


# In[ ]:


#fit Model


# In[26]:


lr.fit(X_train, y_train)


# In[32]:


y_pred= lr.predict(X_train)


# In[29]:


lr.score(X_test, y_test)


# In[33]:


plt.scatter(y_train, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# In[34]:


y_test_pred= lr.predict(X_train)


# In[ ]:


#Predict


# In[36]:


lr.score(X_test, y_test)


# In[ ]:




