#!/usr/bin/env python
# coding: utf-8

# # Empathy assessment from eye fixations
# 

# ## Importing libraries

# In[1]:


import pandas as pd
import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np


# In[10]:


#!pip install pandas


# ## Extracting the data from zip and concatenating them

# In[2]:


path = r'E:\University of Essex\DataScience\CE-888 Assignment 2\DataScience\CE888-2205593\EyeT'   
all_files = glob.glob(os.path.join(path, "*.csv"))  

df_from_each_file = (pd.read_csv(f, low_memory=False) for f in all_files)
concatenated_data   = pd.concat(df_from_each_file, ignore_index=True)


# In[11]:


concatenated_data


# ## Exploring data

# In[12]:


concatenated_data.info()


# In[13]:


concatenated_data.isnull().sum()


# In[14]:


concatenated_data.describe()


# ## Preprocessing data

# In[15]:


concatenated_data = concatenated_data.drop(['Project name'], axis= 1)
concatenated_data = concatenated_data.drop(['Timeline name'], axis= 1)
concatenated_data = concatenated_data.drop(['Recording Fixation filter name'], axis= 1)
concatenated_data = concatenated_data.drop(['Presented Stimulus name'], axis= 1)
concatenated_data = concatenated_data.drop(['Presented Media position X (DACSpx)'], axis= 1)
concatenated_data = concatenated_data.drop(['Presented Media position Y (DACSpx)'], axis= 1)
concatenated_data = concatenated_data.drop(['Mouse position X'], axis= 1)
concatenated_data = concatenated_data.drop(['Mouse position Y'], axis= 1)
concatenated_data = concatenated_data.drop(['Gaze event duration'], axis= 1)


# In[9]:


concatenated_data


# ## Visualizing data

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


plt.figure(figsize=(10,8))
plt.hist(concatenated_data['Eye movement type'], bins=20, color='purple')
plt.title("Distribution of Eye Movement Type")
plt.xlabel("Eye Movement Type")
plt.ylabel("Frequency")
plt.show()


# In[21]:


concatenated_data['Recording date UTC'] = pd.to_datetime(concatenated_data['Recording date UTC'])
concatenated_data.set_index('Recording date UTC', inplace=True)
plt.plot(concatenated_data.index, concatenated_data['Eye movement type index'])


# In[18]:


import plotly.express as px
fig = px.pie(concatenated_data['Eye movement type index'].value_counts().reset_index(), values='Eye movement type index', names='index')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    title_text="Eye movement type index")
fig.show()


# In[20]:


concatenated_data['Sensor']=concatenated_data['Sensor'].astype('category').cat.codes
concatenated_data['Presented Media name']=concatenated_data['Presented Media name'].astype('category').cat.codes
concatenated_data['Recording monitor latency']=concatenated_data['Recording monitor latency'].astype('category').cat.codes
concatenated_data['Fixation point Y (MCSnorm)']=concatenated_data['Fixation point Y (MCSnorm)'].astype('category').cat.codes
concatenated_data['Fixation point X (MCSnorm)']=concatenated_data['Fixation point X (MCSnorm)'].astype('category').cat.codes
concatenated_data['Eye movement type']=concatenated_data['Eye movement type'].astype('category').cat.codes
concatenated_data['Validity left']=concatenated_data['Validity left'].astype('category').cat.codes
concatenated_data['Validity right']=concatenated_data['Validity right'].astype('category').cat.codes
concatenated_data['Event value']=concatenated_data['Event value'].astype('category').cat.codes
concatenated_data['Event']=concatenated_data['Event'].astype('category').cat.codes
concatenated_data['Recording software version']=concatenated_data['Recording software version'].astype('category').cat.codes


# In[23]:


concatenated_data['Recording start time']=concatenated_data['Recording start time'].astype('category').cat.codes
concatenated_data['Eyetracker timestamp']=concatenated_data['Eyetracker timestamp'].astype('category').cat.codes
concatenated_data['Gaze point Y (MCSnorm)']=concatenated_data['Gaze point Y (MCSnorm)'].astype('category').cat.codes
concatenated_data['Eye movement type index']=concatenated_data['Eye movement type index'].astype('category').cat.codes
concatenated_data['Eye movement type index']=concatenated_data['Eye movement type index'].astype('category').cat.codes
concatenated_data['Gaze point left X (DACSmm)']=concatenated_data['Gaze point left X (DACSmm)'].astype('category').cat.codes
concatenated_data['Gaze point left Y (DACSmm)']=concatenated_data['Gaze point left Y (DACSmm)'].astype('category').cat.codes
concatenated_data['Gaze point right X (DACSmm)']=concatenated_data['Gaze point right X (DACSmm)'].astype('category').cat.codes
concatenated_data['Gaze point left X (DACSmm)']=concatenated_data['Gaze point left X (DACSmm)'].astype('category').cat.codes
concatenated_data['Gaze point Y (MCSnorm)']=concatenated_data['Gaze point Y (MCSnorm)'].astype('category').cat.codes
concatenated_data['Gaze point left X (MCSnorm)']=concatenated_data['Gaze point left X (MCSnorm)'].astype('category').cat.codes
concatenated_data['Gaze point left Y (MCSnorm))']=concatenated_data['Gaze point left Y (MCSnorm)'].astype('category').cat.codes
concatenated_data['Gaze point right X (MCSnorm)']=concatenated_data['Gaze point right X (MCSnorm)'].astype('category').cat.codes
concatenated_data['Gaze point right Y (MCSnorm)']=concatenated_data['Gaze point right Y (MCSnorm)'].astype('category').cat.codes     


# In[28]:


concatenated_data['Eye movement type index'].hist(color='green',bins=10,figsize=(8,6))


# In[8]:


concatenated_data.info()


# ## Splitting data

# In[7]:


Featured_columns = ['Recording timestamp', 'Validity left','Validity right' ,'Sensor', 'Recording duration', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)', 'Gaze point right X (MCSnorm)', 'Gaze point right Y (MCSnorm)','Event','Eyetracker timestamp']
X = concatenated_data[Featured_columns]
y = concatenated_data['Eye movement type']


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[34]:


X_train.shape, X_test.shape


# ## Cross validation

# In[35]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble


# In[36]:


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1

for train_index, test_index in kf.split(X, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1


# In[37]:


score = cross_val_score(linear_model.LogisticRegression(random_state= 42), X, y, cv= kf, scoring="accuracy")
print(f'Each fold scores: {score}')
print(f'The average score: {"{:.2f}".format(score.mean())}')


# ## Random forest

# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf_classifier = RandomForestClassifier(n_estimators=100)


# In[5]:


rf_classifier.fit(X_train, y_train)


# In[41]:


y_pred = rf_classifier.predict(X_test)


# In[42]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))


# ## Hyper-parameter tuning

# In[43]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[45]:


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)


# In[ ]:


grid_search.fit(X, y)


# In[ ]:


print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

