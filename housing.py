#!/usr/bin/env python
# coding: utf-8

# # Import required libraries

# In[30]:


import pandas as pd


# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the data

# In[3]:


House = pd.read_csv('USA_Housing.csv')


# # Dataset Exploration and Preprocessing

# In[4]:


House.head()


# In[5]:


House.info()


# In[6]:


House.describe()


# In[7]:


House.columns


# In[8]:


sns.pairplot(House)


# In[9]:


sns.heatmap(House.corr(), annot=True)


# In[22]:


X = House[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
           
y = House['Price']           


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[25]:


X_train


# # Building the Linear Regression model

# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


lm = LinearRegression()


# In[32]:


lm.fit(X_train, y_train)


# In[34]:


import pandas as pd


# In[37]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])


# In[38]:


coeff_df


# In[39]:


predictions = lm.predict(X_test)


# In[40]:


plt.scatter(y_test, predictions)


# In[41]:


sns.distplot((y_test-predictions),bins=50)


# In[ ]:




