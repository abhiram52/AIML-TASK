#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[9]:


sns.histplot(data1['daily'],kde = True,stat='density',)
plt.show()


# observations

# there are no missing values
# 
# the daily column values appears to be right-skewed
# 
# the sunday column values also appear to be right-skewed
# 
# there are 2 outliers in both daily column and also in sunday column as observed from the

# In[10]:


x= data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# In[13]:


data1.corr(numeric_only=True)


# Observations

# Observations on Correlation strength 
# = 
# • The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot 
# 
# • The correlation is strong and postive with Pearson's correlation coefficient of 0.958154

# Fit a Linear Regression Model

# In[14]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[15]:


model1.summary()


# In[ ]:




