#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[11]:


Univ = pd.read_csv("Universities.csv")
univ


# In[13]:


Univ1 = Univ.iloc[:,1:]
Univ1


# In[14]:


Univ1.info()


# In[17]:


Univ1.columns


# In[18]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df
# Scaler.fit_transform(Univ1)


# In[19]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0) # Specify 3 cluster
clusters_new.fit(scaled_Univ_df)


# In[20]:


# Print the cluster labels
clusters_new.labels_


# In[21]:


#Assign cluster to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[22]:


Univ


# In[23]:


Univ.sort_values(by = "clusterid_new")


# In[24]:


# use groupby() to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations:
# - Cluster 2 appers to be top rated universities cluster as the cut off score,Top10,SFRatio parameter mean values are highest
# - Cluster 1 appers to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities

# In[25]:


Univ[Univ['clusterid_new']==0]


# In[26]:


wcss = []
for i in range(1, 20):

    kmeans = KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    #kmeans.fit(Univ1)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Numner of cluster')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




