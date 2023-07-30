#!/usr/bin/env python
# coding: utf-8

# Veri Setim Hakkında Bilgiler
# 
# İris bir çiçek. 1936’da bir bilim insanı bu çiçeğin üç türüne (setosa, versicolor, virginica) ait 50’şer tane, toplamda 150 tane olmak üzere çiçek bulmuş ve hepsinin üst ve alt çiçek yapraklarını ölçmüş. Bu ölçümden dört nitelikli (sepal-length (alt yaprak uzunluğu cm), sepal-with (alt yaprak genişliği cm), pedal-length (üst yaprak genişliği cm), pedal-width (üst yaprak uzunluğu cm)) ve 150 elemanlı bir veri seti elde etmiştir. Bunlar üzerinden yola çıkarak nitelikleri verilen çiçeklerin hangi türden olduğunu bulmaya yarayan bir model oluşturmaya çalışacağım.

# In[1]:


#Gerekli kütüphaneleri ekledim

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


pd.set_option('display.max_rows', 500)


# In[2]:


#Veri setlerimi ekledim

data = pd.read_csv("data_with_nans.csv")


# In[3]:


#Verime genel bir bakış atıyorum.

data.head() 


# In[4]:


data.info() 


# In[5]:


#İstemediğim özelliğin kolununu çıkarıyorum. (axis = 1)

data.drop(labels = "Unnamed: 0",axis = 1, inplace = True) 


# In[6]:


data.head()


# In[7]:


#Verimin değerlerini inceliyorum.

data.describe().T


# In[8]:


#Species'a göre değerleri inceliyorum.

data.groupby("Species").agg(["min","max","std","mean"])


# In[9]:


#NaN değerlerine bakıyorum.

data.isna().sum()


# In[10]:


for column in data.columns[1:-1]:
    
    data[column].fillna(value = data[column].mean(), inplace = True)


# In[11]:


data.isna().sum()


# In[20]:


#Görselleştirme işlemi yapıyorum.

for column in data.columns[1:-1]: 

    data[column].plot(color="#3A0754")
    
    plt.show()


# In[21]:


#Sınıflandırmayı daha net anlamak adına nokta grafik çizdiriorum.

for column in data.columns[1:-1]:
    
    sns.scatterplot(data=data, x="Id", y=column, hue="Species",color="#3A0754")
    plt.show()


# In[14]:


#Aykırı değerleri standart sapma ile buluyorum ve siliyorum.

for column in data.columns[1:-1]:
    
    for spec in data["Species"].unique():
            
        selected_spec = data[data["Species"] == spec]
        selected_column = selected_spec[column]
        
        std = selected_column.std()
        avg = selected_column.mean()
        
        three_sigma_plus = avg + (3 * std)
        three_sigma_minus = avg - (3 * std)
        
        outliers = selected_column[((selected_spec[column] > three_sigma_plus) | 
                                   (selected_spec[column] < three_sigma_minus))].index
        
        data.drop(index = outliers,inplace = True)


# In[15]:


#Aykırı değerlerden sonra veri setimdeki değişiklikleri inceliyorum.

for column in data.columns[1:-1]:
    
    sns.scatterplot(data=data, x="Id", y=column, hue="Species")
    plt.show()


# In[16]:


# IQR ile aykırı değerleri buluyorum ve dropluyorum.

for column in data.columns[1:-1]:
    
    for spec in data["Species"].unique():
            
        selected_spec = data[data["Species"] == spec]
        selected_column = selected_spec[column]
        
        q1 = selected_column.quantile(0.25)
        q3 = selected_column.quantile(0.75)
        
        iqr = q3 - q1
        
        minimum = q1 - (1.5 * iqr)
        maximum = q3 + (1.5 * iqr)
                
        max_index = data[(data["Species"] == spec) & (data[column] > maximum)].index
        min_index = data[(data["Species"] == spec) & (data[column] < minimum)].index
        
        data.drop(index = max_index, inplace = True)
        data.drop(index = min_index, inplace = True)
        
        
        


# In[17]:


#Aykırı değerlerden sonra veri setimdeki değişiklikleri inceliyorum.

for column in data.columns[1:-1]:
    
    sns.scatterplot(data=data, x="Id", y=column, hue="Species")
    plt.show()


# In[18]:


#Modelimi kaydediyorum

data.to_csv("final.csv")

