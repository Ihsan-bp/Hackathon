#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


df=pd.read_csv('dataframe_.csv')


# In[12]:


df.isna().sum()


# In[13]:


# preparation of data
df.dropna(inplace=True)


# In[14]:


#2.exploratory dat analysis
# df.describe() summary statistics


# In[18]:


df.plot.scatter(kind "scatter". x "input", y "output") scatter plot
plt.scatter(df[input])


# In[19]:


# modelling
from sklearn.linear_model import LinearRegression
model=LinearRegression()
x=df[["input"]]
y=df[["output"]]
model.fit(x,y)


# In[20]:


# validation
from sklearn.metrics import mean_squared_error
predictions=model.predict(x)
mse=mean_squared_error(y,predictions)
print("Mean Squared Error:",mse)


# In[21]:


# conclution
print("the equation of the best fit for the data.")
print("output - {0:.2f}+{1:.2f}*input".format(model.intercept_,model.coef_[0]))


# In[25]:


# another modeling
from sklearn.preprocessing import polynomialFeatures


# In[ ]:





# In[ ]:





# In[ ]:




