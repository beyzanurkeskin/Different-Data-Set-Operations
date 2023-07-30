#!/usr/bin/env python
# coding: utf-8

#  

# # Highest Mountains in the Universe 
# 
# 
# 

# # 
# 

# In[1]:


#Import the necessary libraries

import pandas as pd
import numpy as np


#  

# In[2]:


#Import my data

data = pd.read_csv("query (1).csv")


#  

# In[3]:


#Looking at the features in the dataset

data.head()


# In[4]:


data.tail()


#  

# In[5]:


#Reviewing the dataset.

data.shape


# In[6]:


data.info() 


#  

# In[7]:


#Checking for a duplicated data

data[data.duplicated()]


# There is no duplicate data

#  

# In[8]:


#Checking for missing values

data.isna().sum()


# There are 275 null values in "itemDescription" I will drop them

#  

# In[9]:


#Drop null values

data.dropna(inplace=True)


# ### Checking for invalid values

# In[10]:


data["item"].value_counts().sum()


# In[11]:


from matplotlib import pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(40,10))
sns.countplot(x="item", data=data)  


# In[12]:


(data["item"].value_counts() == 1).sum()


# In[13]:


(data["item"].value_counts() == 2).sum()


# In[14]:


(data["item"].value_counts() == 3).sum()


# I think there is no invalid value in this feature.

# # 

# In[15]:


(data["elevation"] < 0).sum()


# In[16]:


(data["elevation"] < 1000).sum()


# I think there is no invalid value in this feature too.

# # 

# In[19]:


data["unitLabel"].value_counts()


# I think there is no invalid value in this feature too.

# # 

# In[63]:


pd.set_option('display.max_rows', 5000)


# In[75]:


data.drop("itemLabel>2")


# In[ ]:




