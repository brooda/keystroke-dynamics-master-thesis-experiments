#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('data.csv')
df = df[df[" PasswordType"] == 1]
del df["Unnamed: 0"]
del df[" Pressure"]
del df[" FingerArea"]
del df[" PasswordType"]
del df[" Hands"]
del df[" DeviceId"]


# In[3]:


df['gravity_x_normalized'] = (df[" gravityX"] -df[" gravityX"].mean())/df[" gravityX"].std()
df['gravity_y_normalized'] = (df[" gravityY"] -df[" gravityY"].mean())/df[" gravityY"].std()
df['gravity_z_normalized'] = (df[" gravityZ"] -df[" gravityZ"].mean())/df[" gravityZ"].std()


# In[4]:


# min / max X/Y for keys
df_by_key = {x: df[df[" Key"] == x] for x in df[" Key"].unique()}

boundaries_of_keys = {}

for key, part_df in df_by_key.items():
    x = part_df[" RawX"]
    y = part_df[" RawY"]

    boundaries_of_keys[key] = (np.min(x), np.max(x), np.min(y), np.max(y))


# In[5]:


def get_relative_x(row):
    key = row[" Key"]
    x = row[" RawX"]

    min_x = boundaries_of_keys[key][0]
    max_x = boundaries_of_keys[key][1]

    return (x - min_x) / (max_x - min_x)
    
def get_relative_y(row):
    key = row[" Key"]
    y = row[" RawY"]

    min_y = boundaries_of_keys[key][2]
    max_y = boundaries_of_keys[key][3]

    return (y - min_y) / (max_y - min_y)

df["relative_x"] = df.apply(get_relative_x, axis=1)
df["relative_y"] = df.apply(get_relative_y, axis=1)


# In[6]:


df_by_users = {x: df[df.UserId == x] for x in df.UserId.unique()}


# In[7]:


data_transformed_in_sequences = []

for user, df in df_by_users.items():
    needed_sequence = [' .', ' t', ' i', ' e', ' 123?', ' 5', ' abc', ' Shift↑', ' R', ' o', ' a', ' n', ' l']
    
    grouped = df.groupby([" SessionId", " Repetition"])
    
    for state, frame in grouped:
        keys = list(frame[" Key"])
        if keys == needed_sequence:
            start_timestamp = list(frame[" DownTime"])[0]
            
            down_times = np.array(frame[" DownTime"]) - start_timestamp
            up_times = np.array(frame[" UpTime"]) - start_timestamp
            
            down_up = []
            
            for d, u in zip(down_times, up_times):
                down_up.append(d)
                down_up.append(u)
            
            down_up = [el / 1000 for el in down_up]
                
            features = {
                "user": user,
                "session": state[0],
                "repetition": state[1],
                "down_up": down_up,
                "raw_x": list(frame[" RawX"]),
                "raw_y": list(frame[" RawY"]),
                "normalized_x": list(frame["relative_x"]),
                "normalized_y": list(frame["relative_y"]),
                "gravity_x": list(frame[" gravityX"]),
                "gravity_y": list(frame[" gravityY"]),
                "gravity_z": list(frame[" gravityZ"]),
                "gravity_x_normalized": list(frame["gravity_x_normalized"]),
                "gravity_y_normalized": list(frame["gravity_y_normalized"]),
                "gravity_z_normalized": list(frame["gravity_z_normalized"])
            }
            
            data_transformed_in_sequences.append(features)

df = pd.DataFrame(data_transformed_in_sequences)
df.to_csv("data_transformed.csv")


# In[8]:


dfs = []

for user, part_by_user in df.groupby("user"):
    sessions_of_user = np.unique(part_by_user.session)
    session_mapping = {}
    
    for i, el in enumerate(sessions_of_user):
        session_mapping[el] = i
        
    part_by_user.session.replace(session_mapping, inplace = True)
    dfs.append(part_by_user)

df = pd.concat(dfs, ignore_index=True)

counts = df.groupby(["user", "session"]).count()
grouped = df.groupby(["user", "session"])
new_df = grouped.filter(lambda x: len(x) > 16)

grouped_by_user = new_df.groupby(["user"])

def check_if_has_3_sessions(df):
    sessions = pd.unique(df["session"])
    return np.array_equal(sessions, [0, 1, 2])
    
new_df = grouped_by_user.filter(check_if_has_3_sessions)
grouped = new_df.groupby(["user", "session"])

def normalize_repetitions(grp):
    grp["repetition"] = np.arange(len(grp))
    return grp

grouped.apply(normalize_repetitions)
new_df = grouped.filter(lambda x: True)
new_df.to_csv("data_transformed.csv")


# In[ ]:





# In[ ]:




