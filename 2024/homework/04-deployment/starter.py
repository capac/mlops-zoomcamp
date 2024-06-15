#!/usr/bin/env python
# coding: utf-8

# # Homework 4

# In[1]:


import pickle
import pandas as pd
import numpy as np
from pathlib import Path


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[3]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[4]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[5]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ## Question 1
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# * 1.24
# * **6.24**
# * 12.28
# * 18.28

# In[6]:


print(f'The standard deviation is {round(np.std(y_pred), 2)}.')


# ## Question 2
# 
# What's the size of the output file?
# 
# * 36M
# * 46M
# * 56M
# * **66M**

# In[7]:


df.head()


# In[8]:


year, month = 2023, 3
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[18]:


df_results = pd.concat([df['ride_id'], pd.DataFrame(y_pred)])


# In[19]:


df_results.to_parquet(
    'output.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[20]:


print(f'''The size of the output file is {round(Path('output.parquet').stat().st_size/(1024*1024), 2)} MB.''')


# In[11]:


df.head()


# ## Q3. Creating the scoring script
# 
# Now let's turn the notebook into a script. Which command you need to execute for that?

# In[ ]:




