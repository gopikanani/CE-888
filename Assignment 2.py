#!/usr/bin/env python
# coding: utf-8

# ###  Import necessary python library

# In[4]:


import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings            
warnings.filterwarnings("ignore")
import plotly.express as px


# In[5]:


pd.set_option('display.max_columns', None) # display dataset with all columns


# ## 1. Loading and exploring the Dataset

# Extracting the data from zip and concatenating them

# In[6]:


path = r'E:\University of Essex\DataScience\CE-888 Assignment 2\DataScience\CE888-2205593\EyeT'   
all_files = glob.glob(os.path.join(path, "*.csv"))  

df_from_each_file = (pd.read_csv(f, low_memory=False) for f in all_files)
concatenated_data   = pd.concat(df_from_each_file, ignore_index=True)


# In[7]:


concatenated_data


# In[8]:


# Load one csv file from dataset
Eye_fix = pd.read_csv("EyeT/EyeT_group_dataset_III_image_name_letter_card_participant_27_trial_3.csv")


# In[9]:


print('--- First 5 rows of the dataset ---')
Eye_fix.head()


# In[10]:


Eye_fix.tail() # Print the last n rows of the dataset


# In[11]:


Eye_fix.columns # Print the column lables of the dataset


# # Exploring data 

# In[12]:


Eye_fix.info()


# In[13]:


concatenated_data.info()


# In[14]:


concatenated_data.isnull().sum()


# In[15]:


concatenated_data.describe()


# ### Explore the dataset by checking its shape, data types, and basic statistics:

# In[16]:


Eye_fix_data=Eye_fix[['Gaze point X', 'Gaze point Y', 'Gaze point left X',
       'Gaze point left Y', 'Gaze point right X', 'Gaze point right Y','Recording resolution height', 
                      'Recording resolution width','Presented Media width',
       'Presented Media height', 'Presented Media position X (DACSpx)',
       'Presented Media position Y (DACSpx)', 'Original Media width',
       'Original Media height', 'Eye movement type', 'Gaze event duration',
       'Eye movement type index', 'Fixation point X', 'Fixation point Y']]


# In[17]:


print('\n--- Shape of the dataset ---')
Eye_fix_data.shape


# In[18]:


print(Eye_fix_data.dtypes)


# ## 2. Data Cleaning and preprocessing

# In[19]:


concatenated_data = concatenated_data.drop(['Project name'], axis= 1)
concatenated_data = concatenated_data.drop(['Timeline name'], axis= 1)
concatenated_data = concatenated_data.drop(['Recording Fixation filter name'], axis= 1)
concatenated_data = concatenated_data.drop(['Presented Stimulus name'], axis= 1)
concatenated_data = concatenated_data.drop(['Presented Media position X (DACSpx)'], axis= 1)
concatenated_data = concatenated_data.drop(['Presented Media position Y (DACSpx)'], axis= 1)
concatenated_data = concatenated_data.drop(['Mouse position X'], axis= 1)
concatenated_data = concatenated_data.drop(['Mouse position Y'], axis= 1)
concatenated_data = concatenated_data.drop(['Gaze event duration'], axis= 1)


# In[20]:


concatenated_data


# ### Check for missing value

# In[21]:


Eye_fix_data.isnull() # checking missing values


# In[22]:


Eye_fix_data.isnull().sum() # Chcek total missing value by variable


# In[23]:


Eye_fix_data.ffill(axis = 0,inplace=True) # Fill missing values in dataset,


# In[24]:


Eye_fix_data.dropna(inplace=True,axis=0) # removes the rows that contains NULL values
print(Eye_fix_data)


# In[25]:


Eye_fix_data.isnull().sum()


# In[26]:


Eye_fix_data.shape


# In[27]:


Eye_fix_data.describe() # return description of the data in the dataset


# In[28]:


Eye_fix_data.columns


# In[29]:


concatenated_data.isnull()


# In[30]:


concatenated_data.isnull().sum()


# ## 3. Data exploration, Visualization and Feature Extraction

# ### Explore the correlation between different signals and the 'Eye movement type' column

# In[31]:


concatenated_data.corr()['Eye movement type index']


# In[32]:


plt.figure(figsize=(10,8))
concatenated_data['Eye movement type'].value_counts().plot(kind='bar', color='purple')
#plt.hist(concatenated_data['Eye movement type'], bins=20, color='purple')
plt.title("Distribution of Eye Movement Type")
plt.xlabel("Eye Movement Type")
plt.ylabel("Frequency")
plt.show()


# In[33]:


concatenated_data.reset_index(inplace=True)
concatenated_data['Recording date UTC'] = pd.to_datetime(concatenated_data['Recording date UTC'], format='%d.%m.%Y')
concatenated_data.set_index('Recording date UTC', inplace=True)
plt.plot(concatenated_data.index, concatenated_data['Eye movement type index'])


# In[34]:


import plotly.express as px
fig = px.pie(concatenated_data['Eye movement type index'].value_counts().reset_index(), values='Eye movement type index', names='index')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(
    title_text="Eye movement type index")
fig.show()


# In[35]:


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


# In[36]:


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


# In[37]:


concatenated_data['Eye movement type index'].hist(color='green',bins=10,figsize=(8,6))


# In[38]:


concatenated_data.info()


# In[39]:


Eye_fix_data.corr()['Eye movement type index']


# In[40]:


Eye_fix_data['Eye movement type'].value_counts().sort_values().plot(kind='bar') # value_counts return counts of unique values.sort_values return sort dataset by specified lable. 


# In[41]:


fig = px.pie(Eye_fix_data, names='Eye movement type', title ='Pie chart of Eye movement type of data')
fig.show()


# # Splitting data

# In[42]:


Featured_columns = ['Recording timestamp', 'Validity left','Validity right' ,'Sensor', 'Recording duration', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)', 'Gaze point right X (MCSnorm)', 'Gaze point right Y (MCSnorm)','Event','Eyetracker timestamp']
X = concatenated_data[Featured_columns]
y = concatenated_data['Eye movement type']


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[44]:


X_train.shape, X_test.shape


# # Cross validation

# In[45]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
from sklearn.linear_model import LogisticRegression
from joblib import parallel_backend


# In[46]:


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1

for train_index, test_index in kf.split(X, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1


# In[47]:


with parallel_backend('multiprocessing', n_jobs=-1):
    scores = cross_val_score(LogisticRegression(random_state=42), X, y, cv=kf, scoring='accuracy')
    
print(f'Each fold scores: {scores}')
print(f'The average score: {"{:.2f}".format(scores.mean())}')


# # Random forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf_classifier = RandomForestClassifier(n_estimators=5)


# In[49]:


rf_classifier.fit(X_train, y_train)


# In[50]:


y_pred = rf_classifier.predict(X_test)


# In[51]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy*100))


# # Hyper-parameter tuning

# In[52]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[53]:


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)


# In[ ]:


grid_search.fit(X, y)


# In[ ]:


print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

