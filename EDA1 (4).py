#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


# Dataframe attributes
print(type(data))
print(data.shape)


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[7]:


data1[data1.duplicated(keep = False)]


# In[8]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[9]:


# Change column names(Rename the columns)
data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[10]:


data.isnull().sum()


# In[11]:


cols = data1.columns
colors = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[12]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[13]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[14]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[15]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[16]:


###detection of outliers in the columns##
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='red', width=2, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='green', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# In[17]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

sns.boxplot(data=data1["Solar"], ax=axes[0], color='orange', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Radiation Levels")

sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='red', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Radiation Levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

plt.show()


# In[18]:


sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin Plot")
plt.show()


# In[19]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert= False)


# In[20]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"],vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[21]:


data1["Ozone"].describe()


# In[22]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# #### observations
# -It is observed that only two outlayers are idenfied using std method
# -in boxplot method more no. of outliers are identified
# -This is becatus the assumption of normality is not satified in this column

# In[23]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Ouilier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# ##observations from Q-Q plot##
# 
# The data does not follow normal distribution as the data points are deviation significantly away the red line
# 
# 
# 

# In[24]:


sns.violinplot(data=data1["Ozone"], color='black')
plt.title("violine plot")


# In[27]:


sns.swarplot(data=data1, x = "Weather", y = "Ozone",color="orange",palette="Set2", size=6)


# In[28]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[29]:


sns.boxplot(data = data1, x = "Weather", y="Ozone")


# In[30]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[31]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[32]:


data1_numeric.corr()


# In[33]:


sns.pairplot(data1_numeric)


# In[35]:


data1_numeric.corr()


# Observations 
# 
# • The highest correlation strength is observed between Ozone and Temperature (0.597087) 
# 
# • The next higher correlation strength is observed between Ozone and wind (-0.523738) 
# The next higher correlation strength is observed between wind and Temp (-0.441228) 
# 
# • The least correlation strength is observed between solar and wind (-0.055874)
# 

# Transformations

# In[39]:


#Creating dummy variable for Weather column
data2=pd.get_dummies(data1,columns=['Month','Weather'])
data2


# Normalization

# In[40]:


data1_numeric.values


# In[43]:


#Normalization of the data 
from numpy import set_printoptions 
from sklearn.preprocessing import MinMaxScaler 

array = data1_numeric.values 

scaler = MinMaxScaler (feature_range=(0,1)) 
rescaledX = scaler.fit_transform(array) 

#transformed data 
set_printoptions(precision=2) 
print(rescaledX [0:10,:])


# In[ ]:




