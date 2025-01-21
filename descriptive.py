#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[10]:


df = pd.read_csv("Universities.csv")
df


# In[11]:


np.std(df["GradRate"])


# In[12]:


np.var(df["SFRatio"])


# In[13]:


df.describe()


# In[ ]:




