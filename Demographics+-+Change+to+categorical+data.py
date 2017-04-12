
# coding: utf-8

# Varisara Tansakul
# vtansaku@vols.utk.edu
# 12 April 2017

# Purpsoe: Changes categorical data in data/hh-demographics.csv to numerical entries
# that are easier to deal with than strings.
# We will now combine this demographic data with the sales spreadsheet to 
# create the final .CSV file we need for analysis

# In[2]:

import pandas as pd
import os


# In[13]:

demo = pd.read_csv('hh_demographic.csv')
demo.head()


# In[15]:

demo["AGE_DESC"][demo["AGE_DESC"] == "19-24"] = 1
demo["AGE_DESC"][demo["AGE_DESC"] == "25-34"] = 2
demo["AGE_DESC"][demo["AGE_DESC"] == "35-44"] = 3
demo["AGE_DESC"][demo["AGE_DESC"] == "45-54"] = 4
demo["AGE_DESC"][demo["AGE_DESC"] == "55-64"] = 5
demo["AGE_DESC"][demo["AGE_DESC"] == "65+"] = 6
        
demo["MARITAL_STATUS_CODE"][demo["MARITAL_STATUS_CODE"] == "A"] = 1
demo["MARITAL_STATUS_CODE"][demo["MARITAL_STATUS_CODE"] == "B"] = 2
demo["MARITAL_STATUS_CODE"][demo["MARITAL_STATUS_CODE"] == "U"] = 3
    
demo["INCOME_DESC"][demo["INCOME_DESC"] == "Under 15K"] = 1
demo["INCOME_DESC"][demo["INCOME_DESC"] == "15-24K"] = 2
demo["INCOME_DESC"][demo["INCOME_DESC"] == "25-34K"] = 3
demo["INCOME_DESC"][demo["INCOME_DESC"] == "35-49K"] = 4
demo["INCOME_DESC"][demo["INCOME_DESC"] == "50-74K"] = 5
demo["INCOME_DESC"][demo["INCOME_DESC"] == "75-99K"] = 6
demo["INCOME_DESC"][demo["INCOME_DESC"] == "100-124K"] = 7
demo["INCOME_DESC"][demo["INCOME_DESC"] == "125-149K"] = 8
demo["INCOME_DESC"][demo["INCOME_DESC"] == "150-174K"] = 9
demo["INCOME_DESC"][demo["INCOME_DESC"] == "175-199K"] = 10
demo["INCOME_DESC"][demo["INCOME_DESC"] == "200-249K"] = 11
demo["INCOME_DESC"][demo["INCOME_DESC"] == "250K+"] = 12

demo["HOMEOWNER_DESC"][demo["HOMEOWNER_DESC"] == "Homeowner"] = 1
demo["HOMEOWNER_DESC"][demo["HOMEOWNER_DESC"] == "Probable Owner"] = 2
demo["HOMEOWNER_DESC"][demo["HOMEOWNER_DESC"] == "Probable Renter"] = 3
demo["HOMEOWNER_DESC"][demo["HOMEOWNER_DESC"] == "Renter"] = 4
demo["HOMEOWNER_DESC"][demo["HOMEOWNER_DESC"] == "Unknown"] = 5
    
demo["HH_COMP_DESC"][demo["HH_COMP_DESC"] == "1 Adult Kids"] = 1
demo["HH_COMP_DESC"][demo["HH_COMP_DESC"] == "2 Adults Kids"] = 2
demo["HH_COMP_DESC"][demo["HH_COMP_DESC"] == "2 Adults No Kids"] = 3
demo["HH_COMP_DESC"][demo["HH_COMP_DESC"] == "Single Female"] = 4
demo["HH_COMP_DESC"][demo["HH_COMP_DESC"] == "Single Male"] = 5
demo["HH_COMP_DESC"][demo["HH_COMP_DESC"] == "Unknown"] = 6

demo["HOUSEHOLD_SIZE_DESC"][demo["HOUSEHOLD_SIZE_DESC"] == "1"] = 1
demo["HOUSEHOLD_SIZE_DESC"][demo["HOUSEHOLD_SIZE_DESC"] == "2"] = 2
demo["HOUSEHOLD_SIZE_DESC"][demo["HOUSEHOLD_SIZE_DESC"] == "3"] = 3
demo["HOUSEHOLD_SIZE_DESC"][demo["HOUSEHOLD_SIZE_DESC"] == "4"] = 4
demo["HOUSEHOLD_SIZE_DESC"][demo["HOUSEHOLD_SIZE_DESC"] == "5+"] = 5

demo["KID_CATEGORY_DESC"][demo["KID_CATEGORY_DESC"] == "None/Unknown"] = 0
demo["KID_CATEGORY_DESC"][demo["KID_CATEGORY_DESC"] == "1"] = 1
demo["KID_CATEGORY_DESC"][demo["KID_CATEGORY_DESC"] == "2"] = 2
demo["KID_CATEGORY_DESC"][demo["KID_CATEGORY_DESC"] == "3+"] = 3


# In[16]:

demo


# In[28]:

demo.to_csv('demo_ToCategorical.csv', index = False)


# In[29]:

print demo.shape


# In[ ]:



