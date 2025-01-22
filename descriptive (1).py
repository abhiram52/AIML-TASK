#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[3]:


np.std(df["GradRate"])


# In[4]:


np.var(df["SFRatio"])


# In[5]:


df.describe()


# In[8]:


#### visualizations###
# visualize the GradRate using histor
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


plt.hist(df["GradRate"])


# In[12]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[ ]:




