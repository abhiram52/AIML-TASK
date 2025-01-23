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


# In[6]:


#### visualizations###
# visualize the GradRate using histor
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


plt.hist(df["GradRate"])


# In[8]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[9]:


s = [20,15,10,25,30,35,28,150,200]
scores = pd.Series(s)
scores


# In[10]:


plt.boxplot(scores,vert=False)


# In[11]:


###identify outliers in universities dataset###


# In[12]:


df = pd.read_csv("universities.csv")
df


# In[14]:


plt.figure(figsize=6,2))
plt.title("Box plot for SAT Score")
plt.boxplot(df["SAT"], vert = False)


# In[ ]:




