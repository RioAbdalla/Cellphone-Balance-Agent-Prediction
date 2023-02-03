#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None



# read the data
df = pd.read_csv('2020.csv',sep=';')

# shape and data types of the data
print(df.shape)
print(df.dtypes)

# select numeric columns
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)

# select non numeric columns
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)


# In[2]:


#turn to numerical for easier manipulation
df['Nilai_UMR'] = pd.to_numeric(df['Nilai_UMR'], errors='coerce')
df['Pengguna_Ponsel'] = pd.to_numeric(df['Pengguna_Ponsel'], errors='coerce')
df['Jumlah_Agen_Pulsa'] = pd.to_numeric(df['Jumlah_Agen_Pulsa'], errors='coerce')


# In[3]:


## sanity check
df.head()


# In[4]:


## check columns
df.columns


# # Cleaning and Wrangling

# ### Missing Values

# #### Initial Check

# In[5]:


#counting missing values
for col in df.columns:
    pct_missing = df[col].isnull().sum()
    print('{} - {}'.format(col,pct_missing))


# #### MISSING VALUES PEMILIK PONSEL(MEDIAN Region)

# In[6]:


## determine Pemilik_Ponsel column for missing values
nan_in_col  = df[df['Pemilik_Ponsel'].isnull()]
print(nan_in_col)


# In[7]:


## filter data for Sulawesi region only
df_sul = pd.DataFrame(df[df['Regional'] == 'Sulawesi'])


# In[8]:


## determine the median of pemilik ponsel in sulawesi region
median_pemilik_ponsel_sulawesi = df_sul['Pemilik_Ponsel'].median()
median_pemilik_ponsel_sulawesi


# In[9]:


df_nan_pemilik_ponsel = pd.DataFrame(nan_in_col)


# In[10]:


## replace the null with median of pemilik ponsel in sulawesi
df.at[297,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
df.at[360,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
df.at[380,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
df.at[407,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
df.at[423,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi
df.at[452,'Pemilik_Ponsel'] = median_pemilik_ponsel_sulawesi


# In[11]:


## sanity check
df[df['Pemilik_Ponsel'].isnull()]


# #### MISSING VALUES PENGGUNA PONSEL

# In[12]:


## Determine the missing values in pengguna ponsel columns
nan_in_col2  = df[df['Pengguna_Ponsel'].isnull()]
print(nan_in_col2)


# In[13]:


## determine the median of pengguna ponsel base on sulawesi region
median_pengguna_ponsel = df_sul['Pengguna_Ponsel'].median()
median_pengguna_ponsel


# In[14]:


## replace the missing values with median of pengguna ponsel in Sulawesi Region
df.at[301,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[312,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[323,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[329,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[332,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[353,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[362,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[363,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[369,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[371,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[375,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[392,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[412,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[416,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[419,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[432,'Pengguna_Ponsel'] = median_pengguna_ponsel
df.at[471,'Pengguna_Ponsel'] = median_pengguna_ponsel


# In[15]:


## sanity check
df[df['Pengguna_Ponsel'].isnull()]


# #### MISSING VALUES JUMLAH AGEN PULSA

# In[16]:


## Determine the missing values in jumlah agen pulsa column
nan_in_col3  = df[df['Jumlah_Agen_Pulsa'].isnull()]
print(nan_in_col3)


# In[17]:


## determine the median of jumlah agen pulsa based on sulawesi region
median_agen_pulsa_sulawesi = df_sul['Jumlah_Agen_Pulsa'].median()


# In[18]:


## filter dataframe for papua barat region
df_papua_barat = pd.DataFrame(df[df['Provinsi'] == 'Papua Barat'])


# In[19]:


## determine the median of jumlah agen pulsa based on papua barat
median_agen_pulsa_papua_barat = df_papua_barat['Jumlah_Agen_Pulsa'].median()


# In[20]:


## replace the missing values with median of jumlah agen pulsa in sulawesi and papua barat
df.at[329,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
df.at[332,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
df.at[344,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
df.at[363,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
df.at[383,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi
df.at[444,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_papua_barat
df.at[451,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_papua_barat
df.at[471,'Jumlah_Agen_Pulsa'] = median_agen_pulsa_sulawesi


# In[21]:


## sanity check
df[df['Jumlah_Agen_Pulsa'].isnull()]


# #### Categorical Missing Values

# In[22]:


#check missing values
df[df.isnull().any(axis=1)]


# In[23]:


## sanity check of provinsi unique values
df['Provinsi'].unique()


# In[24]:


#replace provinsi missing values with the appropriate name
df.at[98,'Provinsi'] = 'Sumatera Utara'
df.at[167,'Provinsi'] = 'Jawa Barat'
df.at[175,'Provinsi'] = 'Jawa Barat'
df.at[235,'Provinsi'] = 'Jawa Tengah'
df.at[237,'Provinsi'] = 'Jawa Tengah'
df.at[261,'Provinsi'] = 'Jawa Timur'
df.at[332,'Provinsi'] = 'Sulawesi Tenggara'
df.at[338,'Provinsi'] = 'Papua Barat'


# In[25]:


#sanity check
for col in df.columns:
    pct_missing = df[col].isnull().sum()
    print('{} - {}'.format(col,pct_missing))


# In[26]:


## sanity check
df[df.isnull().any(axis=1)]


# In[27]:


#missing values of nilai UMR in sumatera utara
df.loc[(df.Provinsi == 'Sumatera Utara') & (df['Nilai_UMR'].isnull())]


# In[28]:


## determine the minimum wage of sumatera utara from secondary sources and replace the missing values with it's suitable values
UMR_SUMUT = 3222526
df.at[14,'Nilai_UMR'] = UMR_SUMUT
df.at[20,'Nilai_UMR'] = UMR_SUMUT
df.at[58,'Nilai_UMR'] = UMR_SUMUT


# In[29]:


#missing values of nilai UMR in sulawesi utara
df.loc[(df.Provinsi == 'Sulawesi Utara') & (df['Nilai_UMR'].isnull())]


# In[30]:


## determine the minimum wage of sulawesi utara from secondary sources and replace the missing values with it's suitable values
UMR_SULUT = 3310000
df.at[311,'Nilai_UMR'] = UMR_SULUT
df.at[326,'Nilai_UMR'] = UMR_SULUT
df.at[327,'Nilai_UMR'] = UMR_SULUT
df.at[331,'Nilai_UMR'] = UMR_SULUT
df.at[333,'Nilai_UMR'] = UMR_SULUT
df.at[335,'Nilai_UMR'] = UMR_SULUT
df.at[337,'Nilai_UMR'] = UMR_SULUT
df.at[340,'Nilai_UMR'] = UMR_SULUT
df.at[342,'Nilai_UMR'] = UMR_SULUT
df.at[351,'Nilai_UMR'] = UMR_SULUT
df.at[352,'Nilai_UMR'] = UMR_SULUT
df.at[355,'Nilai_UMR'] = UMR_SULUT
df.at[376,'Nilai_UMR'] = UMR_SULUT
df.at[399,'Nilai_UMR'] = UMR_SULUT
df.at[404,'Nilai_UMR'] = UMR_SULUT


# In[31]:


#missing values of nilai UMR in Sulawesi Tengah
df.loc[(df.Provinsi == 'Sulawesi Tengah') & (df['Nilai_UMR'].isnull())]


# In[32]:


## determine the minimum wage of sulawesi tenggara from secondary sources and replace the missing values with it's suitable values
UMR_SULTENG = 2390739
df.at[325,'Nilai_UMR'] = UMR_SULTENG
df.at[365,'Nilai_UMR'] = UMR_SULTENG
df.at[378,'Nilai_UMR'] = UMR_SULTENG
df.at[383,'Nilai_UMR'] = UMR_SULTENG
df.at[413,'Nilai_UMR'] = UMR_SULTENG
df.at[426,'Nilai_UMR'] = UMR_SULTENG
df.at[428,'Nilai_UMR'] = UMR_SULTENG
df.at[434,'Nilai_UMR'] = UMR_SULTENG


# In[33]:


#sanity Check
df[df.isnull().any(axis=1)]


# #### Zero Values

# In[34]:


## Determine the number of zero values
for col in df.columns:
    pct_zero = (df[col] == 0).sum()
    print('{} - {}'.format(col,pct_zero))


# In[35]:


## determine the zero numbers in nilai UMR column
df.loc[(df.Nilai_UMR == 0)]


# In[36]:


## determine the minimum wage of all zero values region from secondary sources
UMR_Jambi = 2930000
UMR_Bengkulu = 2215000
UMR_Babel = 3230000
UMR_Jabar = 3700000
UMR_Jateng = 2810000
UMR_Sulsel = 3255000
UMR_Sulbar = 2678000
UMR_Maluku = 2604000
UMR_Kalbar = 2434000
UMR_Kalsel = 2877000
UMR_Papuabarat = 3516000
UMR_Kaltim = 3014000
UMR_NTT = 1975000 
UMR_Jatim = 4375000


# In[37]:


## replace the missing values with it's suitable values
df.at[0,'Nilai_UMR'] = UMR_Jambi
df.at[3,'Nilai_UMR'] = UMR_Bengkulu
df.at[4,'Nilai_UMR'] = UMR_Jambi
df.at[7,'Nilai_UMR'] = UMR_Bengkulu
df.at[17,'Nilai_UMR'] = UMR_Bengkulu
df.at[21,'Nilai_UMR'] = UMR_Jambi
df.at[24,'Nilai_UMR'] = UMR_Bengkulu
df.at[28,'Nilai_UMR'] = UMR_Jambi
df.at[29,'Nilai_UMR'] = UMR_Jambi
df.at[33,'Nilai_UMR'] = UMR_Bengkulu
df.at[42,'Nilai_UMR'] = UMR_Jambi
df.at[46,'Nilai_UMR'] = UMR_Bengkulu
df.at[49,'Nilai_UMR'] = UMR_Bengkulu
df.at[54,'Nilai_UMR'] = UMR_Bengkulu
df.at[56,'Nilai_UMR'] = UMR_Bengkulu
df.at[59,'Nilai_UMR'] = UMR_Jambi
df.at[62,'Nilai_UMR'] = UMR_Bengkulu
df.at[82,'Nilai_UMR'] = UMR_Jambi
df.at[94,'Nilai_UMR'] = UMR_Jambi
df.at[99,'Nilai_UMR'] = UMR_Jambi
df.at[104,'Nilai_UMR'] = UMR_Jambi
df.at[107,'Nilai_UMR'] = UMR_Babel
df.at[109,'Nilai_UMR'] = UMR_Babel
df.at[136,'Nilai_UMR'] = UMR_Babel
df.at[137,'Nilai_UMR'] = UMR_Babel
df.at[139,'Nilai_UMR'] = UMR_Babel
df.at[142,'Nilai_UMR'] = UMR_Babel
df.at[144,'Nilai_UMR'] = UMR_Babel
df.at[171,'Nilai_UMR'] = UMR_Jabar
df.at[207,'Nilai_UMR'] = UMR_Jatim
df.at[212,'Nilai_UMR'] = UMR_Jateng
df.at[236,'Nilai_UMR'] = UMR_Jateng
df.at[266,'Nilai_UMR'] = UMR_Jateng
df.at[364,'Nilai_UMR'] = UMR_Sulsel
df.at[380,'Nilai_UMR'] = UMR_Sulbar
df.at[391,'Nilai_UMR'] = UMR_Maluku
df.at[393,'Nilai_UMR'] = UMR_Kalbar
df.at[405,'Nilai_UMR'] = UMR_Kalsel
df.at[411,'Nilai_UMR'] = UMR_Papuabarat
df.at[456,'Nilai_UMR'] = UMR_Kaltim
df.at[498,'Nilai_UMR'] = UMR_NTT


# In[38]:


## sanity check
df.loc[(df.Nilai_UMR == 0)]


# #### ZERO  values DANA ALOKASI UMUM

# In[39]:


## determine dana alokasi umum zero values
df.loc[(df.Dana_Alokasi_Umum == 0)]


# In[40]:


## filter dana alokasi umum based on eastern part of java
df_jawa2 = pd.DataFrame(df[df['Area'] == 'Area 2'])
median_jawa_area2 = df_jawa2['Dana_Alokasi_Umum'].median()


# In[41]:


median_jawa_area2


# In[42]:


## filter dana alokasi umum based on western part of java
df_jawa3 = pd.DataFrame(df[df['Area'] == 'Area 3'])
median_jawa_area3 = df_jawa3['Dana_Alokasi_Umum'].median()


# In[43]:


median_jawa_area3


# In[44]:


## combine the eastern and western part of java filter dataframe
df_jawa_2_dan_3 = pd.concat([df_jawa2, df_jawa3])


# In[45]:


## determine the median of whole java island
median_jawa_2_dan_3 = df_jawa_2_dan_3['Dana_Alokasi_Umum'].median()
median_jawa_2_dan_3


# In[46]:


## replace the zero values with median of whole java island
df.at[156,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
df.at[160,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
df.at[166,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
df.at[168,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
df.at[176,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3
df.at[192,'Dana_Alokasi_Umum'] = median_jawa_2_dan_3


# In[47]:


# sanity check
df.loc[(df.Dana_Alokasi_Umum == 0)]


# **ZERO JUMLAH PENDUDUK BEKERJA**

# In[48]:


## check zero values of jumlah penduduk bekerja
df.loc[(df.Jumlah_Penduduk_Bekerja == 0)]


# In[49]:


## Determine Bengkulu's jumlah penduduk bekerja from secondary sources and replace the zero values with the suitable values
jumlah_penduduk_bekerja_bengkulu = 1002160
df.at[3,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[7,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[17,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[24,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[33,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[46,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[49,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[54,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[56,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu
df.at[62,'Jumlah_Penduduk_Bekerja'] = jumlah_penduduk_bekerja_bengkulu


# In[50]:


## sanity check
df.loc[(df.Jumlah_Penduduk_Bekerja == 0)]


# In[51]:


## sanity check
for col in df.columns:
    pct_zero = (df[col] == 0).sum()
    print('{} - {}'.format(col,pct_zero))


# In[52]:


## save for future use
df.to_csv('2020cleaned.csv')


# ### OUTLIER Detection

# In[53]:


##new integer type columns dataframe 
dfint=df.select_dtypes(include='int64')


# In[54]:


## new float type columns dataframe
dffloat=df.select_dtypes(include='float64')


# In[55]:


##Box plot of every integer type columns dataframe
for column in dfint:
    plt.figure()
    dfint.boxplot([column])


# In[56]:


## Boxplot of every float type columns dataframe
for column in dffloat:
    plt.figure()
    dffloat.boxplot([column])


# ### Normalize Feature

# In[57]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics


# In[58]:


##distribution plot of PDRB
sns.distplot(df['PDRB'] , fit=norm);


(mu, sigma) = norm.fit(df['PDRB'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

## plot of distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('PDRB')


# In[59]:


## normalize PDRB value with Log function
df['LOG_PDRB'] = np.log1p(df['PDRB'])


# In[60]:


## sanity check
df['LOG_PDRB']


# In[61]:


## distribution plot of Logarithmic PDRB
sns.distplot(df['LOG_PDRB'] , fit=norm);


(mu, sigma) = norm.fit(df['LOG_PDRB'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

## plot of Logarihtmic PDRB distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('LOG PDRB')


# In[62]:


## normalize all feature
df['log_PDRB'] = np.log1p(df['PDRB']) 
df['log_PDRB_Per_Kapita'] = np.log1p(df['PDRB_Per_Kapita'])
df['log_Indeks_Pembangunan_Manusia'] = np.log1p(df['Indeks_Pembangunan_Manusia'])
df['log_Dana_Alokasi_Umum'] = np.log1p(df['Dana_Alokasi_Umum'])
df['log_Pengeluaran_Riil_per_Kapita_ per_Tahun'] = np.log1p(df['Pengeluaran_Riil_per_Kapita_ per_Tahun'])
df['log_Nilai_UMR'] = np.log1p(df['Nilai_UMR'])
df['log_Jumlah_Penduduk_Miskin'] = np.log1p(df['Jumlah_Penduduk_Miskin'])
df['log_Jumlah_Penduduk_Bekerja'] = np.log1p(df['Jumlah_Penduduk_Bekerja'])
df['log_Pengguna_Internet'] = np.log1p(df['Pengguna_Internet'])
df['log_Pemilik_Ponsel'] = np.log1p(df['Pemilik_Ponsel'])
df['log_Pengguna_Ponsel'] = np.log1p(df['Pengguna_Ponsel'])
df['log_Jumlah_Penduduk'] = np.log1p(df['Jumlah_Penduduk'])  


# ### Area Encoding

# In[63]:


## Determine unique values of area
df.Area.unique()


# In[64]:


## Determine Unique values of Regional
df.Regional.unique()


# In[65]:


## encode each area value with numerical value
df["Area_encode"] =df.Area.map({'Area 1':1, 'Area 2':2, 'Area 3':3, 'Area 4':4, 'Area 5':5})


# In[66]:


## encode each regional value with numerical value
df["Regional_Encode"] =df.Regional.map({'Sumbagsel':1, 
                                        'Lampung':2, 
                                        'Sumbagut':3, 
                                        'Sumbagteng':5, 
                                        'Jabar':6, 
                                        'Jabo Inner':7, 
                                        'Jabo Outer':8, 
                                        'Jatim':9, 
                                        'Jateng': 10, 
                                        'Sulawesi':11, 
                                        'Kalimantan':12, 
                                        'Malpua':13, 
                                        'Balnus':14})


# In[67]:


## sanity check
df.Regional_Encode.unique()


# In[68]:


## save cleaned data for future use
df.to_csv('2020cleanedlog.csv')


# In[69]:


## sanity check
df


# In[70]:


## new float type columns dataframe
dffloat=df.select_dtypes(include='float64')


# In[71]:


## boxplot of every float type columns
for column in dffloat:
    plt.figure()
    dffloat.boxplot([column])


# ### DUPLICATE

# In[72]:


## Determine if each columns have duplicate rows
key = ['PDRB', 'PDRB_Per_Kapita', 'Indeks_Pembangunan_Manusia', 'Jumlah_Penduduk', 'Dana_Alokasi_Umum', 'Pengeluaran_Riil_per_Kapita_ per_Tahun', 'Nilai_UMR', 'Jumlah_Penduduk_Miskin', 'Jumlah_Penduduk_Bekerja', 'Pengguna_Internet', 'Pemilik_Ponsel', 'Pengguna_Ponsel', 'Jumlah_Agen_Pulsa']
df_dedupped2 = df.drop_duplicates(subset=key)

print(df.shape)
print(df_dedupped2.shape)


# In[73]:


## Drop duplicate rows
df = df.drop_duplicates(subset=key)


# ### Split the value of kelurahan and desa

# In[ ]:


## split jumlah keluarah desa into two separate features: kelurahan and desa
df[['Kelurahan', 'Desa']] = df['Jumlah_Kelurahan_Desa'].str.split('/', expand=True)


# In[ ]:


## sanity check
df.head()


# ### The number of Kelurahan and Desa in each Kota/Kabupaten

# #### Kelurahan

# In[ ]:


## Check values of kelurahan
df.Kelurahan.unique()


# In[ ]:


#replace '-' with nan
df['Kelurahan'] = df.Kelurahan.replace("-", np.nan)


# In[ ]:


## sanity check
df[df.isnull().any(axis=1)]


# In[ ]:


## Sanity Check
df.Kelurahan.unique()


# #### Desa

# In[ ]:


## Check Desa values
df.Desa.unique()


# In[ ]:


## replacing '-' with nan
df['Desa'] = df.Desa.replace("-", np.nan)


# In[ ]:


## sanity check
df.Desa.unique()


# In[ ]:


## Replace None with nan
df.Desa.fillna(value=np.nan, inplace=True)


# In[ ]:


## Sanity Check
df.Desa.unique()


# #### Replace values of kelurahan and desa

# In[ ]:


#replace nan values in kelurahan with zero
df['Kelurahan'] = df['Kelurahan'].fillna(0)


# In[ ]:


#replace nan values in Desa with zero
df['Desa'] = df['Desa'].fillna(0)


# In[ ]:


#change type of kelurahan and desa columns to numerical for easier processing
df['Kelurahan'] = pd.to_numeric(df['Kelurahan'], errors='coerce')
df['Desa'] = pd.to_numeric(df['Desa'], errors='coerce')


# In[ ]:


## new dataframe for sum groupby based on Kota Kabupaten
dfkabupatenkota = df.groupby(['Kota_Kabupaten']).sum()


# In[ ]:


## Show Kelurahan sum
dfkabupatenkota['Kelurahan']


# In[ ]:


## Show Desa sum
dfkabupatenkota['Desa']


# # Correlation of each futures

# In[ ]:


sns.heatmap(df.corr())


# # Modelling

# ### Linear Regression

# In[ ]:


## Check Data
df.head()


# In[74]:


## Split Data to test and train
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

X = df[['log_PDRB_Per_Kapita', 
        'log_Indeks_Pembangunan_Manusia', 
        'log_PDRB', 'log_Jumlah_Penduduk', 
        'log_Dana_Alokasi_Umum', 
        'log_Pengeluaran_Riil_per_Kapita_ per_Tahun', 
        'log_Nilai_UMR', 'log_Jumlah_Penduduk_Miskin', 
        'log_Jumlah_Penduduk_Bekerja', 
        'log_Pengguna_Internet', 
        'log_Pemilik_Ponsel', 
        'log_Pengguna_Ponsel', 
        'Area_encode', 
        'Regional_Encode']]
y = df['Jumlah_Agen_Pulsa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


## traing data with linear Regression model
model = LinearRegression()
clf = model.fit(X_train, y_train)


# In[ ]:


## Predict using vanilla regression
y_pred = clf.predict(X_test)


# In[ ]:


##determine score
clf.score(X_test, y_test)


# In[ ]:


## Determine Root Mean Squared Error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
rmse


# ### Feature Impotance

# In[ ]:


from matplotlib import pyplot
importance = clf.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[ ]:


# Perform lasso reggresion in x_train and y to find feature importance
lasso=Lasso(alpha=0.001)
modellasso = lasso.fit(X_train,y_train)


# In[ ]:


y_predlasso = modellasso.predict(X_test)


# In[ ]:


## Score of Lasso regression
modellasso.score(X_test, y_test)


# In[ ]:


## New dataframe of feature importance
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=X_train.columns)


# In[ ]:


## Show feature importance values
FI_lasso.sort_values("Feature Importance",ascending=False)


# In[ ]:


## Show feature importance Barchart
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
plt.xticks(rotation=90)
plt.show()


# ### XGBoost Model

# In[82]:


## Load XGBoost Library
from xgboost import XGBRegressor
from sklearn import set_config 


# In[76]:


set_config(print_changed_only=False) 
dtr = XGBRegressor(random_state = 42)
print(dtr)


# In[77]:


## Train Model
dtr.fit(X_train, y_train)


# In[78]:


## Determine Score
score = dtr.score(X_test, y_test)
print("R-squared:", score) 


# In[79]:


## Predict future values
from sklearn.metrics import mean_squared_error
from math import sqrt
ypred = dtr.predict(X_test)


# In[80]:


## Determine Mean Squared Error and Root Mean Squared Error
mse = mean_squared_error(y_test, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0)) 


# In[81]:


## Plot predicted data and test data
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 


# ### Feature Selection

# In[89]:


# calculating different regression metrics

from sklearn.model_selection import GridSearchCV


# In[90]:


from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[84]:


pip install mlxtend


# In[85]:


import joblib


# In[86]:


pip install mlxtend --upgrade --no-deps


# #### Forward and backward Feature Selection

# In[87]:


## Download forward selection library
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[91]:


## Forward Selection
sfs1 = sfs(dtr, k_features=(4, 14),
          forward=True, 
          floating=False, 
          scoring='r2',
          cv=cv)


# In[92]:


## Backward Selection
sfs2 = sfs(dtr, k_features=(4,14),
          forward=False, 
          floating=False, 
          scoring='r2',
          cv=cv)


# In[93]:


##Forward Selection train
sfs1 = sfs1.fit(X_train, y_train)


# In[94]:


## Print Significant Feature
feat_names = list(sfs1.k_feature_names_)
print(feat_names)


# In[95]:


sfs1.k_score_


# In[ ]:


print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)


# In[96]:


print('Selected features:', sfs1.k_feature_idx_)


# In[97]:


# Generate the new subsets based on the selected features
X_train_sfs = sfs1.transform(X_train)
X_test_sfs = sfs1.transform(X_test)


# In[98]:


# Fit the estimator using the new feature subset
# and make a prediction on the test data
dtr.fit(X_train_sfs, y_train)


# In[99]:


y_pred = dtr.predict(X_test_sfs)


# In[100]:


# Compute the accuracy of the prediction
score = dtr.score(X_test_sfs, y_test)
print("R-squared:", score) 


# In[101]:


## Plot the optimum number of features using XGBoost
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


# In[102]:


## Plot prediction data and test data
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, y_pred, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 


# In[103]:


## Backward Train
sfs2 = sfs2.fit(X_train, y_train)


# In[104]:


## print significant features
feat_names = list(sfs2.k_feature_names_)
print(feat_names)


# In[105]:


## Print scores
sfs2.k_score_


# In[106]:


## Plot the optimum number of features using XGBoost
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig = plot_sfs(sfs2.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


# In[107]:


# Generate the new subsets based on the selected features
X_train_sfs2 = sfs2.transform(X_train)
X_test_sfs2 = sfs2.transform(X_test)


# In[108]:


# Fit the estimator using the new feature subset
# and make a prediction on the test data
dtr.fit(X_train_sfs2, y_train)


# In[109]:


## Predict with backward selection features
y_pred2 = dtr.predict(X_test_sfs2)


# In[110]:


# Compute the accuracy of the prediction
score = dtr.score(X_test_sfs2, y_test)
print("R-squared:", score) 


# In[ ]:




