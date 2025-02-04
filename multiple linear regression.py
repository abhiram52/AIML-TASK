#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# ## Description of columns

# MPG:Milege of the car
# 
# HP:Horse power of the car
# 
# VOL:Volume of the car(size)
# 
# SP:Top speed of the car
# 
# WT:Weight of the car(pounds)

# In[ ]:




