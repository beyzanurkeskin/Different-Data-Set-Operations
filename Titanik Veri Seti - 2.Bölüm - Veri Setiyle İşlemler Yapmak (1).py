#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Gerekli kütüphaneleri ekledim

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# In[2]:


#Daha fazla satır görüntülemek için bu ayarlamayı yapıyorum
pd.set_option('display.max_rows', 500)


#  

# In[3]:


#Veri setimi ekledim

data = pd.read_csv("ti-train.csv")

test = pd.read_csv("ti-test.csv")


#   

# In[4]:


#Verime genel bir bakış atıyorum.

data.head() 


#  

# In[5]:


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


# In[6]:


num_cols.remove("PassengerId")


# In[7]:


data.loc[data["Age"] < 1]


# In[8]:


data["Age"] = np.where(data["Age"] < 1 , 0 ,data["Age"])


#  

# # Aykırı Değer Analizi

# In[9]:


#Normal dağılım olanları bulmak için grafik çizdirdim.

for x in num_cols:
    
    plt.title(x)
    b = plt.hist(data[x],bins=100)
    plt.show()
    


# In[10]:


for x in num_cols:
    
    fig = plt.figure(figsize =(10, 5))
    
    plt.title(x)
    b = sns.kdeplot(data[x],shade=True,alpha = 0.9,color="black")
    plt.show()


# In[11]:


#Aykırı değerleri bulmak için grafik çizdirdim.

for x in num_cols:
    
    fig = plt.figure(figsize =(10, 5))
    
    plt.title(x)
    b = sns.boxplot(data[x])
    plt.show()


# Histogramlara baktığımda Age normal dağılım göstermiş gibi duruyor.
# 

#  

# In[12]:


data_copy = data.copy() # kopyasını oluşturdum


#  

# In[13]:


#Normal dağılım gösteren ve göstermeyen özelliklerden bir liste oluşturdum.

nor_dis = ["Age"]
not_dis = ["Fare"]


#  

# Normal dağılım gösterenler için standart sapma ve z score yöntemleriyle aykırı değer tespiti yaptım.

# In[14]:


def ss(dataframe, col):

    mean = dataframe[col].mean()
    
    st = dataframe[col].std()
    
    up = mean + 3 * st
    
    low = mean - 3 * st
    
    ss_out = dataframe[(dataframe[col] < low) | (dataframe[col] > up)]
    
    return ss_out


# In[15]:


def z(dataframe, col):
    
    mean = dataframe[col].mean()
    
    st = dataframe[col].std()
    
    zs = (dataframe[col] - mean) / st
    
    z_out = dataframe[(3.5 < zs) | (-3.5 > zs)]
    
    return z_out
    


#  

# Normal dağılım göstermeyenler için iqr ve percentage yöntemleriyle aykırı değer tespiti yaptım.

# In[16]:


def iqr(dataframe, col_name, q1=0.05, q3=0.95):

    quantile1 = dataframe[col_name].quantile(q1)
    
    quantile3 = dataframe[col_name].quantile(q3)
    
    interquantile_range = quantile3 - quantile1
    
    up_limit = quantile3 + 1.5 * interquantile_range
    
    low_limit = quantile1 - 1.5 * interquantile_range
    
    iqr_out = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]
    
    return iqr_out


# In[17]:


def per(dataframe, col_name, q1=0.05, q3=0.95):

    low = dataframe[col_name].quantile(q1)
    
    up = dataframe[col_name].quantile(q3)
    
    per_out = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]
    
    return per_out


#  

# In[18]:


for i in nor_dis:
    out  = ss(data_copy,i)
    print(f"{i}   outliers_num= {len(out)}")


# In[19]:


for i in nor_dis:
    out  = z(data_copy,i)
    print(f"{i}   outliers_num= {len(out)}")


#  

# Zscore ve standart sapma değerleri farklı buldu.

# # 

# In[20]:


for i in not_dis:
    out  = iqr(data_copy,i)
    print(f"{i}   outliers_num= {len(out)}")


# In[21]:


for i in not_dis:
    out  = per(data_copy,i)
    print(f"{i}   outliers_num= {len(out)}")


#  

# IIQR ve percentage da değerleri farklı buldu.

#  

#  

# In[22]:


#Tekrar histogram çizip aykırı değerlere ne uygulayacağıma bakıyorum.

fig = plt.figure(figsize =(20, 10))

plt.title("Age")
sns.boxplot(data["Age"])


# Age değişkeninin standart sapma methoduyla 2 aykırı değeri olduğunu gördüm bu da 73ten büyük değerler demek oluyor ve onları 73e eşitliyorum.

# In[23]:


(data_copy["Age"] > 73).sum()


# In[24]:


(data_copy["Age"] > 73).sum()
data_copy["Age"] = np.where(data_copy["Age"] > 73, 73 ,data_copy["Age"])
(data_copy["Age"] > 73).sum()


# In[25]:


(data_copy["Age"] > 73).sum()


# # 

# In[26]:


fig = plt.figure(figsize =(20, 10))

plt.title("Fare")
sns.boxplot(data["Fare"])


# Fare değişkeninin IQR methoduyla 3 aykırı değeri olduğunu gördüm bu da 300den büyük değerler demek oluyor ve onları 300e eşitliyorum.

# In[27]:


(data_copy["Fare"] > 300).sum()


# In[28]:


data_copy["Fare"] = np.where(data_copy["Fare"] > 300, 300 ,data_copy["Fare"])


# In[29]:


(data_copy["Fare"] > 300).sum()


# # 

# In[30]:


#Aykırı değerleri değerlendirmeden önceki değerleri yazdırdım.

data.describe().T


# In[31]:


#Aykırı değerleri değerlendirdikten sonraki değerleri yazdırdım.

data_copy.describe().T


# İyileşme gördüğüm için orijinal veri setime bu işlemleri doğrudan aktarıyorum.

# In[32]:


data["Age"] = np.where(data["Age"] > 73, 73 ,data["Age"])


# In[33]:


data["Fare"] = np.where(data["Fare"] > 300, 300 ,data["Fare"])


#  

# # 

# In[34]:


#Kategorik kolonlar için aykırı değer bulma işlemlerine başlıyorum.

for n in cat_cols:
    
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x=n, data=data_copy)    


# Sadece cabin için bakacağım outlier var mı diye.

# In[35]:


#Cabin değişkenini inceledim.

fig = plt.figure(figsize=(40,10))
sns.countplot(x="Cabin", data=data_copy)  


# In[36]:


(data_copy["Cabin"].value_counts().unique())


# In[37]:


(data_copy["Cabin"].value_counts() == 1).sum()


# In[38]:


(data_copy["Cabin"].value_counts() == 2).sum()


# In[39]:


(data_copy["Cabin"].value_counts() == 3).sum()


# In[40]:


(data_copy["Cabin"].value_counts() == 4).sum()


# Doluluk oranı çok az, kendi içinde çok fazla ayrışıyor. Bu nedenle şimdilik bir işlem yapmaya gerek duymuyorum.

# # 

# # Eksik Değer Analizi

# In[41]:


#Eksik değerleri buluyorum.

data_copy.isna().sum()


# Embarked değişkeninde çok az eksik değer olduğundan onu mode ile dolduracağım.

# In[42]:


data["Embarked"].mode()


# In[43]:


data["Embarked"] = data["Embarked"].fillna("S")


#  

# In[44]:


#Yaş değişkenini inceledim.

fig = plt.figure(figsize=(40,10))
sns.countplot(x="Age", data=data_copy)  


# In[45]:


#KNN uygulamak için gerekli kütüphaneleri ekliyorum.

from sklearn.impute import KNNImputer


# In[46]:


cat_variables = data_copy[["Sex", "Embarked"]]
cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
cat_dummies.head()


# In[47]:


data_copy = data_copy.drop(['Sex', 'Embarked',"Cabin","PassengerId","Name","Ticket"], axis=1)
data_copy = pd.concat([data_copy, cat_dummies], axis=1)
data_copy.head()


# In[48]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_copy = pd.DataFrame(scaler.fit_transform(data_copy), columns = data_copy.columns)
data_copy.head()


# In[49]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
data_copy = pd.DataFrame(imputer.fit_transform(data_copy),columns = data_copy.columns)


# In[50]:


data_copy.isna().sum()


# In[51]:


data_copy = pd.DataFrame(scaler.inverse_transform(data_copy),columns = data_copy.columns)


# In[52]:


#Eksik değerlerden önce

data["Age"].describe().T


# In[53]:


#Eksik değerlerden sonra

data_copy["Age"].describe().T


# In[54]:


plt.title("age")
b = plt.hist(data_copy["Age"],bins=100,color="black")
b = plt.hist(data["Age"],bins=100,alpha= 0.7,color="yellow")
plt.show()
 


# In[55]:


data["new_age"] = data_copy["Age"]


# In[56]:


fig = plt.figure(figsize =(10, 5))
    
plt.title("age")
b = sns.distplot(data_copy["Age"],color="orange")
b = sns.distplot(data["Age"],color="black")
plt.show()   


# # 

# # 

# # Yeniden Şekillendirmeler 

# In[57]:


# Cabin değerinde çok fazla boş değer var( %78 ) bu nedenle onunla farklı bir işlem yapacağım.

data["Cabin"] = data["Cabin"].isnull().apply(lambda x: not x)

for i in data["Cabin"]:
    
    if i == True:
        data["Cabin"] = data["Cabin"].replace(True,1) 
    
    elif i == False:
        data["Cabin"] = data["Cabin"].replace(False,0) 
        


# # 

# In[58]:


corr_pairs = data.corr().unstack()                          
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
positive_pairs = sorted_pairs[sorted_pairs > 0.4 ]      
print(positive_pairs)


# Yüksek korele olan özellik göremiyorum bu nedenle herhangi bir özelliği silmeyeceğim

# In[59]:


data = data.copy()


# In[60]:


data.drop(["PassengerId","Ticket"], axis=1,inplace=True)


# In[61]:


(data["Fare"] >= 40).sum()


# In[62]:


((data["Fare"] >= 30) & (data["Fare"] < 40)).sum()


# In[63]:


((data["Fare"] >= 20) & (data["Fare"] < 30)).sum()


# In[64]:


((data["Fare"] >= 10) & (data["Fare"] < 20)).sum()


# In[65]:


(data["Fare"] < 10).sum()


# In[74]:


for i in data["Fare"]:
    
    print(i)
    
    if i >= 40:
        print("--------------",i)
        
        data["Fare2"] = 5
    else:
        data["Fare2"] = 1
        
    """
    elif i >= 30 and i< 40 :
        data["Fare"] = data["Fare"].replace(i,4)
        
    elif i >= 20 and i< 30 :
        data["Fare"] = data["Fare"].replace(i,3) 
        
    elif i >= 10 and i< 20 :
        data["Fare"] = data["Fare"].replace(i,2)
        
    else:
        data["Fare"] = data["Fare"].replace(i,1) 
     """   #apply pd 


# In[75]:


data.head()


# In[68]:


fig = plt.figure(figsize=(50,20))
sns.countplot(x="Fare", data=data)  


# # 

# # Modele Hazır Hale Getirme

# In[69]:


data.to_csv("final-titanic.csv")

