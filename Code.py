# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:49:23 2019

@author: Heisenberg
"""
import pandas as pd
import numpy as np

test=pd.read_csv('Test_u94Q5KV.txt')

train=pd.read_csv('Train_UWu5bXk.txt')
sample_submission=pd.read_csv('SampleSubmission_TmnO39y.txt')
#describes columns with int values stats
test.apply(lambda x: sum(x.isnull()))
train.apply(lambda x: sum(x.isnull()))

test['Item_Weight'].fillna(np.mean(test['Item_Weight']),inplace=True)
train['Item_Weight'].fillna(np.mean(train['Item_Weight']),inplace=True)

test['Outlet_Size'].fillna('Medium',inplace = True)
train['Outlet_Size'].fillna('Medium',inplace = True)

test['Item_Fat_Content'].replace('LF','Low Fat', inplace = True)
test['Item_Fat_Content'].replace('reg','Regular', inplace = True)
test['Item_Fat_Content'].replace('low fat','Low Fat', inplace = True)

train['Item_Fat_Content'].replace('LF','Low Fat', inplace = True)
train['Item_Fat_Content'].replace('reg','Regular', inplace = True)
train['Item_Fat_Content'].replace('low fat','Low Fat', inplace = True)

#test['Item_Type'].unique()
#test['Item_Fat_Content'].unique()
#test['Outlet_Size'].unique()
#test['Outlet_Location_Type'].unique()
#test['Outlet_Type'].unique()

#train['Item_Type'].unique()
#train['Item_Fat_Content'].unique()
#train['Outlet_Size'].unique()
#train['Outlet_Location_Type'].unique() 
#train['Outlet_Type'].unique()
del test['Outlet_Establishment_Year']
del test['Item_Identifier']
del test['Outlet_Identifier']

del train['Outlet_Establishment_Year']
del train['Item_Identifier']
del train['Outlet_Identifier']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
train['Item_Fat_Content'] = labelencoder_x.fit_transform(train['Item_Fat_Content'])
train['Item_Type'] = labelencoder_x.fit_transform(train['Item_Type'])
train['Outlet_Size'] = labelencoder_x.fit_transform(train['Outlet_Size'])
train['Outlet_Location_Type'] = labelencoder_x.fit_transform(train['Outlet_Location_Type'])
train['Outlet_Type'] = labelencoder_x.fit_transform(train['Outlet_Type'])

test['Item_Fat_Content'] = labelencoder_x.fit_transform(test['Item_Fat_Content'])
test['Item_Type'] = labelencoder_x.fit_transform(test['Item_Type'])
test['Outlet_Size'] = labelencoder_x.fit_transform(test['Outlet_Size'])
test['Outlet_Location_Type'] = labelencoder_x.fit_transform(test['Outlet_Location_Type'])
test['Outlet_Type'] = labelencoder_x.fit_transform(test['Outlet_Type'])       


#model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train.iloc[:,:-1], train.iloc[:,-1])



# Predicting the Test set results
y_pred = regressor.predict(test)
y_pred=pd.DataFrame(y_pred)
y_pred.to_csv("predicted_value.csv")
sample_submission.to_csv("submission.csv")
