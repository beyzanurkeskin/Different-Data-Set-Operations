#!/usr/bin/env python
# coding: utf-8

# #### 7 yıl boyunca ABD'deki 100.000 bina için enerji kullanım yoğunluğu hesaplanmka istemiştir. 
# #### Her satır bir binayı temsil eder ve belli başlı kategoriler şöyledir; 
# 
# ##### year factor : binaya 7 yıl içinde hangi yıl bakıldığı
# ##### state factor: binanın bulunduğu durum
# ##### building class: bina sınıflandırması
# ##### facility type: bina kullanım tipi
# ##### floor area: binanın taban alanı (feet kare olarak)
# ##### year built: binanın yapıldığı yıl
# ##### .
# ##### .
# ##### .
# ##### ve sıcaklık gibi dış etkenler
# 
# #### Bizim eğitim setiyle oluşturacağımız binaların eui'ını tahmin eden modelimiz, bize sunulmayan
# #### test seti ile değerlendirilecektir.

# # First describe and explain the problem and dataset (explain the columns). Then show your analysis step by step.

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)

dft = pd.read_csv("train.csv") # bize verilen eğitim setini yüklüyoruz


dft.head() # eğitim setini tanımak için ilk 5 satırı yazdırıyoruz


# In[2]:


dfs = pd.read_csv("test.csv") # test setini yüklüyoruz
dfs.head( )# test setini tanımak için ilk 5 satırı yazdırıyoruz


# beklendiği üzere site_eui verileri burda yok o nedenle 63 sütun gözlemliyoruz


# # 

# ## Check the duplicated data in train and test sets

# In[3]:


dft[dft.duplicated()]  #duplicated data kontrolü yapıyoruz ve sonuç her iki set için de sıfır çıkıyor


# In[4]:


dfs[dfs.duplicated()]


# # 

# ## Plot and interpret the correlation table for numerical values

# In[5]:


#bu fonksiyonla birlikte numeric, categoric ve cardinal sütunları belirledim
#aynı zamanda numeric görünen ama categoric olan (mesela Year_Factor) ve 
#categoric ama cardinal (mesela facility_type) olanları da belirledim


def grab_col_names(dataframe, cat_th = 10, car_th = 200):
    
    
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "bool", "object"]]
    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and 
                   str(dataframe[col].dtypes) in ["category", "bool", "object"]]
    
    cat_cols = cat_cols + num_but_cat
    
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    
    return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dft)

num_cols = [col for col in num_cols if col != "id"] #bence id çıkarılmalı çünkü cardinal 

df = dft[num_cols]

df.corr().style.background_gradient().set_precision(2)


# In[8]:


"""

cat_cols:

['Year_Factor',
 'State_Factor',
 'building_class',
 'facility_type',
 'days_above_110F',
 'direction_peak_wind_speed']

Bunları bir kontrol edelim gerçekten kategorikler mi diye.

"""
fig = plt.figure(figsize=(16,6))
sns.countplot(x='Year_Factor', data=dft)


# In[71]:


fig = plt.figure(figsize=(16,6))
sns.countplot(x='State_Factor', data=dft)


# In[72]:


fig = plt.figure(figsize=(16,6))
sns.countplot(x='building_class', data=dft)


# In[19]:


fig = plt.figure(figsize=(20,10))
sns.countplot(x='days_above_110F', data=dft)

#şüpheli görünüyor o yüzden bunu da kategorik listesinden çıkarıyorum

cat_cols.remove("days_above_110F")

num_cols.append("days_above_110F")


# In[10]:


fig = plt.figure(figsize=(16,6))
sns.countplot(x='direction_peak_wind_speed', data=dft)

#Bize verilen açıklamaya göre bunu kategorik kabul edemeyiz o yüzden cat_cols'tan çıkarıyoruz.

cat_cols.remove("direction_peak_wind_speed")

num_cols.append("direction_peak_wind_speed")


# In[11]:


import seaborn as sns

fig = plt.figure(figsize=(40,40))
sns.heatmap(df.corr())
plt.show()


# In[12]:


df.corr()["site_eui"].abs().nlargest(11) #hedef değişkenimizi etkileyen 10 özelliği belirledim 


# In[13]:


df.corr()["site_eui"].nlargest(11) #yukarda mutlak değer almıştık burda ise pozitif etkisi olanlara eriştik


# # 

# ## Plot percentage of data belonging to both Train and Test data for state factor feature and explain.

# In[14]:


a = dft["State_Factor"].value_counts(normalize=True) * 100

a.plot.pie(figsize=(5,5))

c = pd.DataFrame(a)
c.style


# # test setinde büyük çoğunlukta yeri olan state6 faktörünü train setinde hiç göremiyoruz ama burdan ne sonuç çıkarabiliriz bilemedim

# In[15]:


b = dfs["State_Factor"].value_counts(normalize=True) * 100

b.plot.pie(figsize=(5,5))
d = pd.DataFrame(b)
d.style


# ## Plot the countplot graphic for distribution of buildings constructed w.r.t each year and add your interprets.

# Bina inşa yılları 1920lerden sonraki yaklaşık 10 yıllık dönemde zirveye ulaşmış bunun sonucu 1.dünya savaşı sonrası nüfusu artırmaya yönelik bir çalışma olarak görülebilir, böylelikle 2. dünya savaşı sırasında (1940lar) bina inşa sayısındaki keskin düşüş de açıklanabilir. Sonrasında savaşların durması ile nüfus artırma politikaları da yavaşlamış ve inşa edilen bina sayıları önceki yıllara göre daha orantılı şekilde değişmiştir.

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x ='year_built', data = dft.loc[dft['year_built'] > 1920])
sns.set(rc={'figure.figsize':(60,15)})
plt.show()


# ### Create a table for showing all features’ total entry, missing value number,  missing value percentage, data type,  unique values, unique values number

# Bu kısım için eğer kolonları gezme işlemini fonksiyon içinde yapıp daha sonra dataframe döndürürsen istediğimiz sonucu elde ederiz. Bununla ilgili biraz çalış eğer yapamazsan bize yazarsan yardımcı oluruz.

# In[18]:


def tb(df):
    
    l = []
    
    for col in dft.columns:
        l.append(dft[col].unique().tolist())
        
        

    data = {'total_entry':len(df), 
            'missing_value_number':df.isnull().sum().values.tolist(), 
            'missing_value_percentage':((dft.isnull().sum()/len(df)) * 100).tolist(), 
            'data_type': df.dtypes.values.tolist(),
            'unique_values': l,
            'unique_values_number': df.nunique().tolist()}
    
    

    dataframe = pd.DataFrame(data, index=[df.columns.tolist()])
    
    

    return dataframe


tb(dft)


# # 

# ## Define categorical-nominal variables as 'object' type, since the differences within variables do not have meaning (Ex. Nominal, Cardinal, Continuous)

# https://www.mygreatlearning.com/blog/types-of-data/ bu websitesini incelemeni ve daha sonra bunlarla ilgili detaylı araştırma yaparak bizim özelliklerimizin hangi veri türlerinde olduğunu belirlemeni istiyoruz. 

# In[20]:


for s in cat_cols:
    
    dft[s] = dft[s].astype(object)
    
for i in cat_cols:
    
    print(i,"--->",dft[i].dtypes)


# ## Nominal data: 
# 
# Kategorik değişken olup herhangi bir sıraya/ üstünlüğe sahip olmayan verilerdir. Futbol takımları buna örnek verileilir.
# 
# ## Ordinal data:
# 
# Kategorik değişken olup bir sıraya/ üstünlüğe sahip olan verilerdir. Eğitim düzeyi buna örnek verilebilir.
# 
# 
# ## Discrete data:
# 
# Nümerik olup tam sayı tipinde olan verilerdir. Bir fabrikadaki işçi sayısı buna örnek verilebilir.
# 
# 
# ## Continuous data:
# 
# Nümerik olup kesirli sayı tipinde olan verilerdir. İnsan boyu buna örnek verilebilir.

# # 

# ## What is Outlier Detection? 

# Aykırı değer verideki genel eğilimin dışına çıkan değerdir. Mesela lise son sınıftaki öğrencilerinin yaşlarının 17-18-19  olması normaldir ancak 20+ veya 17- olması bir aykırı değer olarak kabul edilebilir. outlier detecton da verideki bu aykırılıklarısaptamak için kullandığımız metotlardır.

# # 

# ## How many Outlier Detection methods are there?  
# 
# 

# --Standart sapma 
# 
# --Z-score
# 
# --IQR
# 
# --LOF:2.png)

# # 

# ## Research 3 of them  and write a documentary. (Separate for categorical and numeric)

# Standart sapma
# 
# Verinin ortalaması ve standart sapması belirlenir. Ortalamaya standart sapma eklenir ve çıkarılır. Örneğin tereyağ fiyatlarını topladık. Ortalama 50₺ standart sapmamız da 30 olsun. Bu verilerle markete gittiğimizde 17 liralık tereyağdan şüphe ederiz çünkü 17 bizim için aykırı bi değer. Aynı şekilde 90 liralık bir tereyap görüp işin içinde bir dolandırıcılık var da diyebiliriz. 
# 
# 
# --
# 
# IQR
# 
# Bir sınıfın notları elimizde olsun. %25lik kısmı(Q1) ve %75lik kısmı(Q3) hesaplarız. Q1 = 25 ve Q3 = 75 diyelim. Q3-Q1 = IQR değerimiz yani 50. Eşik değerimizi de alttan Q1-1.5 x IQR = -50 üstten Q3 + 1.5 x IQR = 150 olarak belirleriz. Bu işlemlerimize göre -50 nin altında ve 150 nin üstünde kalanlar bizim için outlier değerler olur.
# 
# 
# --
# 
# Z-Score
# 
# Verilerimizin z-skorlarının mutlak değerini hesaplar ve mutlak değerce 2 veya 3’ten yüksek olan gözlemleri aykırı değer olarak belirleyebiliriz. Z skorunu  x = bizim değişkenimiz, o = ortalama, s = standar sapma kabul edersek, (x-o) / s şeklinde hesaplayabiliriz.

# # 

# ## How can we handle the missing values for features? 

# --Silinebilir.
# 
# --Belirlnen bir değeri atanabilir.
# 
# --Tahmine dayalı değer atama yapılabilir.
# 

# # 

# ## Research missing value imputation methods and write a documentary.   (Separate for categorical and numeric) 

# Silmek
# 
# Eğer veri setimizde eksik veriler varsa bunları direk silebiliriz. Ancak böylesi işlemler yapmak, eğer eksik değerler çoksa, yanıltıcı sonuçlar doğurabilir. 
# 
# --
# 
# Belirli Bir Değer Atamak
# 
# Eksik değerler yerine ortalama bir değer hesaplayıp tüm boş değerlere bu ortalama değeri yazabiliriz. Basit atama yöntemlerindendir. 
# 
# 
# --
# 
# Tahmine Dayalı Değer Atamak
# 
# İstatistiksel bazı yöntemlerle veya makine öğrenmesiyle bir değer hesaplanıp boş yerlere atanabilir. Daha gelişmiş bir yöntemdir.

# # 
# 

# ## Print the mean, std, min, %25, %50, %75, max values before implementing and after implementing the methods to compare the change of the distribution 

# ## Plot the box plot and distplot of the data before and after implementing the methods to compare the change of the distribution  

# # 

# #  1.SORU - Burada ben yine grafiklere bakarak kendim karar verdim hangisi normal dağılmış hangisi dağılmamış diye. Yaren Hocam standart sapma 1 ortalaması 0 olacak demişti ama onu pek anlayamadım. Araştırdığımda anlaşılır bir şey de bulamadım.

# In[214]:


for x in num_cols:
    
    plt.title(x)
    b = plt.hist(dft[x],bins=750)
    plt.show()
    


# In[270]:


for x in num_cols:
    
    fig = plt.figure(figsize =(20, 15))
    
    plt.title(x)
    b = sns.boxplot(dft[x])
    plt.show()


# bunlara baktığımda 
# 
# site_eui
# 
# ELEVATION
# 
# year_built
# 
# floor_area
# 
# 
# normal dağılım göstermiş gibi geldi

# # 

# In[473]:


dfc = dft.copy() # kopyasını oluşturdum


# In[474]:


nor_dis = ["site_eui","ELEVATION","year_built","floor_area"] #kendime göre normal dağılanları belirledim ve ayırdım

not_dis = [col for col in num_cols if col not in nor_dis]


# # 

# ## Implement the following methods separately for each numeric column to find the outlier points:
# If the data is normally distributed:
# 
# Implement standard deviation and z-score (both of them) 

# In[475]:


from statistics import mode  #outlierları bulduğum zaman o değerlerin modu ile dolduracağım 


# In[548]:


def ss(dataframe, col):
    
    #burada önce standart sapma metodunu uyguladım 

    mean = dataframe[col].mean()
    
    st = dataframe[col].std()
    
    up = mean + 3 * st
    
    low = mean - 3 * st
    
    mode = dataframe[col].mode()
                    
    dataframe.loc[((dataframe[col] > up) | (dataframe[col] < low))] = np.nan #outlierları önce boş değer yaptım
    
    dataframe[col] = dataframe[col].fillna(mode[0]) #sonra bu boş değerleri modela doldurdum
    
    
                    
    
    


# In[477]:


def z(dataframe, col):
    
    #burada önce z-score metodunu uyguladım 
    
    mean = dataframe[col].mean()
    
    st = dataframe[col].std()
    
    zs = (dataframe[col] - mean) / st
    
    dataframe.loc[((zs > 3) | (zs < -3))] = np.nan
    
    
    
    
    
    


# In[189]:


for col in nor_dis:
    
    outliers = ss(dft,col)
    
    #burada önceki kodumda return etmiştim onu sildim geri getireceğim ama burda sayılar gözümün önünde 
    #olsun diye koda ellemedim bu kodu çalıştırırsanız çalışmaycaktır, sadece bana yol gösteriyor
    
    print(f"{col} ---> aykırı_değer_sayısı ---> {len(outliers)}")


# In[188]:


for col in nor_dis:
    outliers = z(dft,col)
    
    #burası da aynı şekilde 
    
    print(f"{col} ---> aykırı_değer_sayısı ---> {len(outliers)}")


# # If the data is NOT normally distributed:
# 
# Implement interquartile range and percentage methods (both of them)

# In[520]:


def iqr(dataframe, col_name, q1=0.05, q3=0.95):

    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    
    #iqr uygulamak için gereken işlemleri yaptım z score ve ss te yaptıklarımla aynı şekilde outlierları mode ile doldurdum
    
    mode = dataframe[col].mode()
    
    
    
    dataframe.loc[((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit))] = np.nan
    
    dataframe[col] = dataframe[col].fillna(mode[0])
    

    
    


# In[479]:


def per(dataframe, col_name, q1=0.01, q3=0.99):

    low = dataframe[col_name].quantile(q1)
    up = dataframe[col_name].quantile(q3)

    dataframe.loc[((dataframe[col] > up) | (dataframe[col] < low))] = np.nan
    
    


# In[304]:


for col in not_dis:
    outliers = iqr(dft,col)
    print(f"{col} ---> aykırı_değer_sayısı ---> {len(outliers)}\n")
    
    #burası da bana yol göstermesi için


# # 

# In[315]:


for col in not_dis:
    outliers = per(dft,col)
    print(f"{col} ---> aykırı_değer_sayısı ---> {len(outliers)}\n")
    
    #burası da bana yol göstermesi için


# In[549]:


for col in nor_dis:
    
    ss(dfc,col)
    
    #sadece ss ve iqr da modla doldurma işlemlerini yaptığım için şimdi de onları çağırdım


# In[528]:


for col in not_dis:
    
    iqr(dfc,col)


# In[4]:


dft.describe().T #önceki datasetin değerleri


# In[554]:


dfc.describe().T #kopya datasetin değerleri


# ## After implementing the above methods, observe the results and explain. (Compare them). Are the same points found as outliers with both methods?

# z score ve ss aynı değerleri verdi ancak iqr ve per oldukça farklı değerler üretti 
# 
# bana yıol gösteren ve çalışmayan kod parçalarından bu sonucu çıakrdım orayı düzeltmeye çalışacağım sadece
# çözemediğim soruları daha hızlı sormak için onlarla uğraşmadım şimdilik

# # 2.SORU - ben boş değerlerin hepsini modela doldurduğum halde neden kopya verisetimdeki count değerlerinin hepsi 75757 değil ?

# # 

# ## Implement the method for each categorical column to find the outlier points:

# ## Plot bar chart or histogram to see the percentage of availability of each category.

# In[283]:


for n in cat_cols:
    
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x=n, data=dft)    


# state factor için state10 ya hiç yok ya da çok çok az bu nedenle onu outlier olarak kabul ediyorum
# 
# building class zaten 2 farklı değişkenden oluşuyor onda bir outlier bulunmuyor
# 
# year factorde 1'i outlier olarak kabul ediyorum.

# # 3.SORU bunlarla ( ^ ) ne yapacağımı bilemedim z score veya ss gibi bir metot bulamadım

# # 

# ## Find Columns with Missing Values

# Implement most appropriate 2 different techniques for each column to handle the missing values.
# 
# Bu tekniklerden biri silme biri de belirli bir değer atamadır.
# 
# Explain, why this technique is appropriate for the column.
# 
# Daha az olanları silip, daha çok olanlara ortalama bir değer atanabilir.

# In[562]:


dfc3 = dfc.copy()


# # 4.SORU - burdan sonra ne zaman kodu çalıştırsam değişik değişik şeyler çıkıyor yeni kopyalar üretip sıfırdan deniyorum anlamadığım sayılar çıkıyor neden olduğu konusunda bir fikir sahibi olamıyorum. Üstelik bir de normal dağılım göstermeyenlerde mode ile doldurma metodum işe yararken normal dağılım gösterenelrded işe yaramıyor ona da bakabilir misiniz?

# In[ ]:





# In[563]:


dfc3.isnull().sum()


# direction_max_wind_speed      
# direction_peak_wind_speed     
# max_wind_speed                
# days_with_fog 
# energy_star_rating 
# 
# 
# bunlar çok fazla missing value içeren featurelar o yüzden bunlara ortalamalarını atayacağız

# In[567]:


more = ["direction_max_wind_speed","direction_peak_wind_speed","max_wind_speed","days_with_fog","energy_star_rating"]
less = [col for col in dfc2.columns if col not in nor_dis] 


# In[568]:


for col in more:
    
    mode = dfc3[col].mode()
    
    dfc3[col] = dfc3[col].fillna(mode[0])


# In[570]:


dfc3.isnull().sum()


# In[ ]:




