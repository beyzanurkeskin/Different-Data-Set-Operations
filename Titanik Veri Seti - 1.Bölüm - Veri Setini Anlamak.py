#!/usr/bin/env python
# coding: utf-8

#  

# # 

# # Titanik Veri Seti

# #### Veri setim Titanik adlı gemideki yolcuların bazı bilgilerini içeriyor. Buna göre ben de bu verilerler bir model kurarak bana verilecek yeni yolcu bilgilerinden,yolcunun yaşama şansının kaç olduğunu tahmin etmeye çalışacağım.
# 
# Özellikler şu şekilde:
# 
#  0   PassengerId - Yolcu numarası                                                             
#  
#  1   Survived    - Hayatta kalma durumu
#  
#  2   Pclass      - Yolculuk sınıfı
#  
#  3   Name        - İsim
#  
#  4   Sex         - Cinsiyet
#  
#  5   Age         - Yaş
#  
#  6   SibSp       - Kardeş/eş sayısı
#  
#  7   Parch       - Ebeveyn/çocuk sayısı
#  
#  8   Ticket      - Bilet numarası
#  
#  9   Fare        - Bilet ücreti
#  
#  10  Cabin       - Kabini
#  
#  11  Embarked    - Bindiği liman
# 

# # 

# In[8]:


#Gerekli kütüphaneleri ekledim

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# In[9]:


#Daha fazla satır görüntülemek için bu ayarlamayı yapıyorum
pd.set_option('display.max_rows', 500)


#  

# In[10]:


#Veri setimi ekledim

data = pd.read_csv("ti-train.csv")

test = pd.read_csv("ti-test.csv")


#   

# In[11]:


#Verime genel bir bakış atıyorum.

data.head() 


#  

# In[12]:


#Verimi inceliyorum.

data.shape


# 12 sütun bulunuyor.
# 

#  

# In[13]:


data.info() 


# 7 nümerik 5 kategorik sütun barındırıyor.
# 

#  

# In[14]:


#Boş değerlere bakıyorum

data.isna().sum()


# Pek fazla boş değer yok gibi görünüyor.

#  

# In[15]:


#Sadece boş değer olanları yazdırıyorum.

NaCol = [col for col in data.columns if data[col].isnull().sum() > 0]
naCol = data[NaCol].isnull().sum()

print(naCol)


#  

# In[33]:


# Boş değerlerin oranlarına bakmak için grafik çizdiriyorum

col = NaCol
nan = list(naCol)


plt.xlabel("col names")
plt.ylabel("nan values")

plt.xticks(fontsize = 15)

plt.bar(col,nan,color= "#9B474E")
plt.show()


#  

# In[17]:


def col_names(dataframe, cat_th = 10, car_th = 200):
    
    
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "bool", "object"]]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and 
                   str(dataframe[col].dtypes) in ["category", "bool", "object"]]
    
    cat_cols = cat_cols + num_but_cat
    
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = col_names(data)


# In[18]:


num_cols


# In[19]:


cat_cols


# In[20]:


cat_but_car


# Nümerik olarak passenger id gösteriliyor ancak onun cardinal değişken olduğunu biliyorum bu nedenle siliyorum.

# In[21]:


num_cols.remove("PassengerId")


#  

# In[22]:


#Nümerik kolonların değerlerine bakıyorum.

data.describe().T


#  

#  

# In[23]:


#Target value için histogram grafiği çizdiriyorum.

fig = plt.figure(figsize = (4, 4))

plt.hist(data["Survived"],bins=3,color="#9B474E",range=[0,1])

plt.show()


#  

# In[24]:


#Kopya veri var mı diye kontrol ediyorum.

data[data.duplicated()]


# Yokmuş

#  

# In[25]:


# Korelasyona bakıyorum.

data.corr().style.background_gradient().set_precision(2)


# In[26]:


# Heatmap ile bakıyorum.

fig = plt.figure(figsize=(5,5))
sns.heatmap(data.corr())
plt.show()


# In[27]:


#Hedef değişkenini etkileyen 5 özelliği belirledim.

data.corr()["Survived"].abs().nlargest(6) 


# In[28]:


#Yukarda mutlak değer almıştım burda ise pozitif etkisi olanlara baktım.

data.corr()["Survived"].nlargest(6) 


#  

#  

# In[29]:


# Veri setimin genle bir incelemesini yapıyorum

def tb(df):
    
    l = []
    t = []
    
    for col in data.columns:
        l.append(data[col].unique().tolist())
        t.append(len(data) - data[col].isna().sum())
        
        

    dat = {'total_entry': t,
            'missing_value_number':df.isnull().sum().values.tolist(), 
            'missing_value_percentage':((df.isnull().sum()/len(df)) * 100).tolist(), 
            'data_type': df.dtypes.values.tolist(),
            'unique_values': l,
            'unique_values_number': df.nunique().tolist()}
    
    

    dataframe = pd.DataFrame(dat, index=[df.columns.tolist()])
    
    

    return dataframe


tb(data)


#  

# 
# Özelliklerin hangi veri tipinde olduğunu bulmaya çalıştım.
# 

# In[30]:


data.select_dtypes("int64").columns


# In[31]:


data.select_dtypes("float64").columns


# In[32]:


data.select_dtypes("O").columns


# PassengerId  -> Cardinal
#                  
# Survived     -> Nominal
# 
# Pclass       -> Ordinal
# 
# Name         -> Cardinal
# 
# Sex          -> Nominal
# 
# Age          -> Continuous
# 
# SibSp        -> Ordinal
# 
# Parch        -> Ordinal
# 
# Ticket       -> Cardinal
# 
# Fare         -> Continuous
# 
# Cabin        -> Nominal
#  
# Embarked     -> Nominal

#  
