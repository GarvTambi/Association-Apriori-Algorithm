#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Association Rule Mining via Apriori Algorithm in python


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


# In[4]:


store_data = pd.read_csv("/home/garv/Desktop/store_data.csv")


# In[5]:


store_data.head()


# In[6]:


# In this dataset there is no header row. But by default, pd.read_csv function 
# treats first row as header. To get rid of this problem, add header=None option to
# pd.read_csv function,as shown below:


# In[7]:


store_data = pd.read_csv("/home/garv/Desktop/store_data.csv", header = None)


# In[10]:


store_data.head()


# In[13]:


store_data.shape


# In[15]:


# Data Preprocessing
#  To convert our pandas dataframe into a list of lists, execute the following script:

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])


# In[28]:


association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)


# In[29]:


# We convert the rules found by the apriori class into a list since it is easier to view the results in this form.
association_results = list(association_rules)


# In[30]:


# Viewing the Results

print(len(association_results))


# In[31]:


print(association_results[0])


# In[33]:


for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")


# In[ ]:




