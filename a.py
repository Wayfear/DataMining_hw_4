
# coding: utf-8

# In[18]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split  
from math import radians, cos, sin, asin, sqrt


# In[19]:


def link(row):
    return str(int(row[0])) + str(int(row[1]))

    data = pd.read_csv("data/final_2g_tr.csv")
test_data = pd.read_csv("data/final_2g_te.csv")
public_data = pd.read_csv("data/final_2g_gongcan.csv")

column = [2,3,13,14,15,16,17,18,23]

train_select_column = data.iloc[:, column]
test_select_column = test_data.iloc[:,column]
all_select_column = train_select_column.append(test_select_column)
useful_public_data = public_data.iloc[:, [5,6,13,14]]
print train_select_column

print useful_public_data

useful_public_data.apply(link)

# In[26]:





# In[ ]:

result_column = [48]
train_result_column = data.iloc[:,result_column]
test_result_column = test_data.iloc[:,result_column]
all_result_column = train_result_column.append(test_result_column)

print all_result_column


# In[ ]:




# In[ ]:

train_X, test_X, train_Y, test_Y = train_test_split(all_select_column, all_result_column, test_size=0.2)


# In[ ]:



