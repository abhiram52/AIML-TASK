#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[27]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[28]:


cars = pd.DataFrame(cars, columns=["HP","VOL","WT","MPG"])
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

# In[29]:


cars.info()


# In[30]:


cars.isna().sum()


# In[31]:


cars


# ### Observations about info(),missing values
# There are no missing values
# 
# There are 81 observations(81 diffrent cars data)
# 
# The data types of the columns are also relvant and valid

# In[32]:


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

# In[33]:


cars[cars.duplicated()]


# In[34]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# ### Observations from correlation plots and Coeffcients 
# • Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG 
# 
# • Therefore this dataset qualifies for building a multiple linear regression model to predict MPG 
# 
# • Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP, VOL vs WT 
# 
# • The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# In[ ]:


model1 = smf.ols('MPG~WT+VOL+SP+HP' ,data=cars).fit()


# In[37]:


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


# In[25]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[23]:


from sklearn.matrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# checking for multicollinearity among X-columns using VIF method

# In[39]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ## Observations for VIF values: 
# • The ideal range of VIF values shall be between 0 to 10. However slightly higher values can be tolerated 
# • As seen from the very high VIF values for VOL and WT, it is clear that they are prone to multicollinearity proble. 
# • Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicollinearity. 
# • It is decided to drop WT and retain VOL column in further models

# In[38]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[ ]:


model2 = smf.ols('MPG~HP+VOL+SP',data=cars1)
model2.summary()


# Performance metrics for model2

# In[41]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[ ]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[ ]:


mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# ## Observations from model2 summary() 
# • The adjusted R-suared value improved slightly to 0.76 
# • All the p-values for model parameters are less than 5% hence they are significant 
# • Therefore the HP, VOL, SP columns are finalized as the significant predictor for the MPG response variable 
# • There is no improvement in MSE value

# In[ ]:




