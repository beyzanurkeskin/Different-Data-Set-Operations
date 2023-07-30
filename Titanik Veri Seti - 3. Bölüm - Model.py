#!/usr/bin/env python
# coding: utf-8

# # 

# In[1]:


#Gerekli kütüphaneleri import ediyorum

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as mns
from sklearn import set_config
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit


# In[2]:


# Hazırladığım veri setimi okuyorum.

data = pd.read_csv("final-titanic.csv")


# In[3]:


# Gerekli görmediğim sütunları siliyorum.

data.drop(["Unnamed: 0"], axis=1,inplace=True)
data.drop(["Name","title","Embarked_Q","Embarked_S"], axis=1,inplace=True)
data.drop(["Age"], axis=1,inplace=True)


# In[4]:


data = data.rename(columns={'new_age': 'Age'})  


# In[5]:


X = data.drop("Survived",axis=1)
y = data["Survived"]

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42,stratify=y)

categorical_processor = ColumnTransformer(transformers=[
    ("OHE",OneHotEncoder(drop='first'),["Sex","Embarked"]),
],remainder="passthrough")

pipe = Pipeline(steps=[
    ("Categorical_Processor",categorical_processor),
    ("Standard Scaling",StandardScaler()),
    ("Model",LogisticRegression())
])
pipe.fit(X_train,y_train)


# In[6]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,pipe.predict(X_test))


# In[7]:


sns.heatmap(confusion_matrix(y_test,pipe.predict(X_test)),annot=True)

plt.show()

