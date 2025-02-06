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

# ### Assumptions in Multilinear Regression 
# 1. Linearity: The relationship between the predictors(X) and the response (Y) is linear. 
# 2. Independence: Observations are independent of each other. 
# 3. Homoscedasticity: The residuals (Y-Y_hat) exhibit constant variance at all levels of the predictor. 
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed. 
# 5. No multicollinearity: The independent variables should not be too highly correlated with each other

# ### EDA

# In[3]:


cars.info()


# In[4]:


cars.isna().sum()


# ### Observations about info(),missing values
# There are no missing values
# 
# There are 81 observations(81 diffrent cars data)
# 
# The data types of the columns are also relvant and valid

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns



fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})


sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')  


sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')  

plt.tight_layout()


plt.show()


# #### Observations from boxplot and histograms 
# • There are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
# 
# • In VOL and WT columns, a few outliers are observed in both tails of their distributions. 
# 
# • The extreme values of cars data may have come from the specially nature of cars
# 
# • As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression   model
# 

# In[6]:


cars[cars.duplicated()]


# In[7]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[8]:


cars.corr()


# ### Observations from correlation plots and Coeffcients 
# • Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG 
# 
# • Therefore this dataset qualifies for building a multiple linear regression model to predict MPG 
# 
# • Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP, VOL vs WT 
# 
# • The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# ### Preparing a preliminary model considering all X columns

# In[16]:


#Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP' ,data=cars).fit()
model1.summary()


# ## Observations from model summary 
# • The R-squared and adjusted R-suared values are good and about 75% of variability in Y is explained by X columns
# 
# • The probability value with respect to F-statistic is close to zero, indicating that all or someof X columns are significant 
# 
# • The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored

# ## Performance metrics for model1

# In[21]:


df1 = pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[23]:


# Predict for given x data columns

pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[ ]:




