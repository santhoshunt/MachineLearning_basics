#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset

# In[3]:


dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[4]:


print(x)


# In[ ]:


print(y)


# # Taking care of missing data

# In[4]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# In[7]:


print(x)


# # Encoding categorical data

# ## Encoding Independent variable

# In[5]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
x = np.array(ct.fit_transform(x))


# In[9]:


print(x)


# ## Encoding Dependent variables

# In[6]:


from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
y = lb.fit_transform(y)


# In[12]:


print(y)


# # Spliting Dataset into testing and training set

# In[28]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[29]:


print(x_train, x_test, y_train, y_test, sep="\n")


# # Feature Scaling
#

# In[32]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])


# In[33]:


print(x_train)


# In[34]:


print(x_test)
