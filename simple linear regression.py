#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[7]:


data1.info()


# In[16]:


data1[data1.duplicate(keep=False)]


# In[15]:


plt.scatter(data1["daily"],data1["sunday"])


# In[ ]:




