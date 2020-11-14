#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_csv("./DSL-StrongPasswordData.csv")

def getNumberOfClass(className):
    subject_names = data.subject.unique()
    return np.where(subject_names == className)[0][0]

data.subject = data.subject.apply(getNumberOfClass) 
data = data.rename(columns={"subject": "user", "sessionIndex": "session", "rep": "repetition"})
data["session"] = data["session"] - 1
data["repetition"] = data["repetition"] - 1 
data.head()


# In[ ]:


strokes = data[['H.period', 'DD.period.t', 'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 'UD.i.e',        'H.e', 'DD.e.five', 'UD.e.five', 'H.five', 'DD.five.Shift.r',        'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o', 'H.o',        'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 'DD.n.l',        'UD.n.l', 'H.l']]

arr = np.array(strokes.values.tolist())

def convert_to_key_down_key_up(row):
    t = 0
    ret = [0]
    
    for i, val in enumerate(row):
        if i%3==0:
            t += val
            ret.append(t)
        if i%3==1:
            t += val - row[i-1]
            ret.append(t)
    return str(ret)

arr = np.array(list(map(convert_to_key_down_key_up, arr)))
data["down_up"] = arr
data = data[["user", "session", "repetition", "down_up"]]
data.to_csv("data_transformed.csv", index=False)


# In[ ]:





# In[ ]:




