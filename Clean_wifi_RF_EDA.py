#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:40:51 2019

@author: eliassmouni
"""

# =============================================================================
# WiFi task
# =============================================================================
# =============================================================================
# Load packages
# =============================================================================
import os

print("Current Working Directory " , os.getcwd())


try:
# Change the current working Directory    
  os.chdir(os.path.expanduser("~/Documents/Ubiqum/wifi task"))
  print("Directory changed")
except OSError:
  print("Can't change the Current Working Directory")        

#necessary Libraries
import numpy as np
import pandas as pd
import time
import pprint
import scipy
from math import sqrt
from numpy import array

#Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import plotly.express as px
from plotly.offline import plot

# other
from sklearn.externals import joblib
import statsmodels.stats.api as sms
from sklearn.utils import resample

#Preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.stats

#Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#Scoring Metrics
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import classification_report

#Cross Validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

# =============================================================================
# DATASETS
# =============================================================================

# import datasets
path = "file:///Users/eliassmouni/Documents/Ubiqum/wifi%20task/Data/trainingData.csv"
trainingData = pd.read_csv(path)

pathval = "file:///Users/eliassmouni/Documents/Ubiqum/wifi%20task/Data/validationData.csv"
validationData = pd.read_csv(pathval)

pathtest = "file:///Users/eliassmouni/Documents/Ubiqum/wifi%20task/Data/testData.csv"
testData = pd.read_csv(pathtest)

# =============================================================================
# DATA EXPLORATION
# =============================================================================

# Work on trainingData first
# Check the structure of the data after it's loaded 
#(print the number of rows and columns).
num_rows, num_cols  = trainingData.shape
print('Number of columns: {}'.format(num_cols))
print('Number of rows: {}'.format(num_rows))

#check the statistics of the data per columns
decription = trainingData.describe()
trainingData.info

#Check the columns names
col_names = trainingData.columns.values
col_names

#check for missing values
missing_values_count = trainingData.isnull().sum()
missing_values_count_test = testData.isnull().sum()

#see the count of missing data per column:
missing_values_count

# how many total missing values do we have?
total_cells = np.product(trainingData.shape)
total_missing = missing_values_count.sum()
total_missing #0

missing_values_count.sum()

# percent of data that is missing
missing_percent = (total_missing/total_cells) * 100
print('Percent of missing data = {}%'.format(missing_percent))

# accesss unique values per feature
## trainingData
unique_floors = trainingData["FLOOR"].unique()
unique_bldgs = trainingData["BUILDINGID"].unique()
unique_spaceid = trainingData["SPACEID"].unique()
unique_rpos = trainingData["RELATIVEPOSITION"].unique()
unique_users = trainingData["USERID"].unique()
unique_phones = trainingData["PHONEID"].unique()

print("Unique Floors : {}".format(unique_floors))
print("Unique Buildings : {}".format(unique_bldgs))
print("Unique Relative Positions : {}".format(unique_rpos))
print("Unique Users : {}".format(unique_users))
print("Unique Space IDs : {}".format(unique_spaceid))
print("Unique Phones : {}".format(unique_phones))

unique_floors_count = len(unique_floors)
unique_bldgs_count = len(unique_bldgs)
unique_spaceid_count = len(unique_spaceid)
unique_rpos_count = len(unique_rpos)
unique_users_count = len(unique_users)
unique_phones_count = len(unique_phones)

## repeat on validationData

#print the number of rows and columns
num_rows_Val, num_cols_Val  = validationData.shape
print('VAL: Number of columns: {}'.format(num_cols_Val))
print('VAL: Number of rows: {}'.format(num_rows_Val))

#check the statistics of the data per columns
decription_Val = validationData.describe()
validationData.info

#Check the columns names
col_names_Val = validationData.columns.values

#check for missing values
missing_values_count_Val = validationData.isnull().sum()

#see the count of missing data per column:
missing_values_count_Val

# how many total missing values do we have?
total_cells_Val = np.product(validationData.shape)
total_missing_Val = missing_values_count_Val.sum()
total_missing_Val #0

# access unique values per feature
unique_floors_val = validationData["FLOOR"].unique()
unique_bldgs_val = validationData["BUILDINGID"].unique()
unique_spaceid_val = validationData["SPACEID"].unique()
unique_rpos_val = validationData["RELATIVEPOSITION"].unique()
unique_users_val = validationData["USERID"].unique()
unique_phones_val = validationData["PHONEID"].unique()

print("Unique Floors VAL : {}".format(unique_floors_val))
print("Unique Buildings VAL : {}".format(unique_bldgs_val))
print("Unique Relative Positions VAL : {}".format(unique_rpos_val))
print("Unique Users VAL : {}".format(unique_users_val))
print("Unique Space IDs VAL : {}".format(unique_spaceid_val))
print("Unique Phones VAL : {}".format(unique_phones_val))

unique_floors_val_count = len(unique_floors_val)
unique_bldgs_val_count = len(unique_bldgs_val)
unique_spaceid_val_count = len(unique_spaceid_val)
unique_rpos_val_count = len(unique_rpos_val)
unique_users_val_count = len(unique_users_val)
unique_phones_val_count = len(unique_phones_val)

### repeat on testdata

#print the number of rows and columns
num_rows_Test, num_cols_Test  = testData.shape
print('TEST: Number of columns: {}'.format(num_cols_Test))
print('TEST: Number of rows: {}'.format(num_rows_Test))

#check the statistics of the data per columns
decription_Test = testData.describe()
testData.info

#Check the columns names
col_names_Test = testData.columns.values

#check for missing values
missing_values_count_Test = testData.isnull().sum()

#see the count of missing data per column:
missing_values_count_Test

# how many total missing values do we have?
total_cells_Test = np.product(testData.shape)
total_missing_Test = missing_values_count_Test.sum()
total_missing_Test #0

# Access unique values of the features
unique_floors_test = testData["FLOOR"].unique()
unique_bldgs_test = testData["BUILDINGID"].unique()
unique_spaceid_test = testData["SPACEID"].unique()
unique_rpos_test = testData["RELATIVEPOSITION"].unique()
unique_users_test = testData["USERID"].unique()
unique_phones_test = testData["PHONEID"].unique()

unique_floors_test_count = len(unique_floors_test)
unique_bldgs_test_count = len(unique_bldgs_test)
unique_spaceid_test_count = len(unique_spaceid_test)
unique_rpos_test_count = len(unique_rpos_test)
unique_users_test_count = len(unique_users_test)
unique_phones_test_count = len(unique_phones_test)

# =============================================================================
# VISUALIZATION
# =============================================================================

# map the data ie. see the campus
# trainingData
trainingData.plot(kind = "scatter", x = "LONGITUDE", y = "LATITUDE", alpha = 0.2)
plt.savefig("data_map.png")

# validationData
validationData.plot(kind = "scatter", x = "LONGITUDE", y = "LATITUDE", alpha = 0.2)
plt.savefig("Val_data_map.png")

# map the data by user ID
# to see how much of the data in each building was collected by how many users
# trainingData
trainingData.plot(kind = "scatter", x = "LONGITUDE", y = "LATITUDE", alpha = 0.4,
                  figsize = (10,7), c = "USERID", cmap = plt.get_cmap("jet"),
                  colorbar = True, sharex = False)
plt.savefig("users_plot.png")
# we can see that not all users walked around in all buildings. Consider discarding the user id
# due to the intuitively obvious bias
#validationData doesn't have user IDs (all are 0)


### plot correlations between the WAP features in the training set
corr_matrix = trainingData.corr()
fig = plt.figure(figsize = (15,15))
sns.heatmap(corr_matrix, xticklabels = False, yticklabels = False)

### plot histograms of the attributes
# trainingData
trainingData.iloc[:, 520:529].hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
# most attributes seem to have a multinomial distribution.
# Long and Lat which seem to have a skewed bell shaped distribution.
# given this complexity -> classify Building and floor IDs
validationData.iloc[:, 520:529].hist(bins=50, figsize=(20,15))
plt.savefig("VAL:attribute_histogram_plots")
# distributions of floor and building are quite different in the validation where the subjects have been allowed to move freely

### scatter matrices
# trainingData
attributes =["BUILDINGID", "FLOOR", "LATITUDE","LONGITUDE", "SPACEID", "RELATIVEPOSITION"]
scatter_matrix(trainingData[attributes], figsize = (12,8))
plt.savefig("matrix.png")
# no linear relationships between these attributes

# validationData
scatter_matrix(validationData[attributes], figsize = (12,8))
plt.savefig("val_matrix.png")
# no linrel

### make 3D plots of the entire campus
# trainingData
fig = px.scatter_3d(trainingData, x = 'LONGITUDE', y = 'LATITUDE', z = 'FLOOR', color = "BUILDINGID")
plot(fig)

# validationData
fig = px.scatter_3d(validationData, x = 'LONGITUDE', y = 'LATITUDE', z = 'FLOOR', color = "BUILDINGID")
plot(fig)
# the training set is not complete, on the fifth floor there an entire corner missing
# the material says that the 520 waps are distributed between the datasets:
# Active in both Training and Validation: 312
# Active in Training alone: 153
# Active in Validation alone: 55
# consider shuffling the training and validation sets
# it will result in a loss of ability to test the model but the trade-off is having a full map so should be worth it
# we can try to make up the lack in measures to test the generalization ability of the model with bootstrap validation
# return to this later

# =============================================================================
# PREPROCESSING 1
# =============================================================================
# PREPROCESSING #
# create separate functions for different tasks so that you can choose which ones to use
# in the different iterations

# first the basic cleaning operation

def clean_data(df):
    """
    Perform conversion of the RSSI signals so that they are on a scale from 0-105
    105 being the strongest signal (the original 0) and 0 being the original 100 (no signal)
    INPUT: trainingData DataFrame
    OUTPUT: Trimmed and cleaned trainingData DataFrame
    """
    # Reverse the values. 100 = 0 and the values range from 0-105 (weak to strong)
    # The intensity values are represented as negative ranging from -104dBm (very weak) to
    # 0dbM
    # the made-up positive value of 100 is used to denote when a WAP was not detected
    df.iloc[:, 0:520] = np.where(df.iloc[:, 0:520] <= 0,
           df.iloc[:, 0:520] + 105,
           df.iloc[:, 0:520] - 100)
    #return the cleaned dataframe
    return df

def drop_cols(df):
    columns_removed = ["USERID", "PHONEID", "TIMESTAMP"]
    for col in columns_removed:
        df.drop(col, axis = 1, inplace =True)
    return df

# define functions to preprocess data for all labels/classes/targets
# Building
def preprocess_data_building(df):
   """
   separates trainingData into independents and dependents
   will also be applied to validationData
   INPUT: Cleaned trainingData dataframe
   OUTPUT: trainingData as independent and dependent variables
   """
   global X
   global y
   # split data into features and targets for another iteration
   X = df.drop(["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION"], axis = 1)
   y = df[["BUILDINGID"]]
   return X, y

# Floor
def preprocess_alldata_floor(df):
   """
   separates trainingData into independents and dependents
   will also be applied to validationData
   INPUT: Cleaned "alldata" dataframe
   OUTPUT: allData as independent and dependent variables
   """
   global X
   global y
   # split data into features and targets for another iteration
   X = df.drop(["LONGITUDE", "LATITUDE", "FLOOR", "SPACEID", "RELATIVEPOSITION"], axis = 1)
   y = df[["FLOOR"]]
   return X, y   

# Longitude
def preprocess_alldata_longitude(df):
   """
   separates trainingData into independents and dependents
   will also be applied to validationData
   INPUT: Cleaned alldata dataframe
   OUTPUT: allData as independent and dependent variables
   """
   global X
   global y
   # split data into features and targets for another iteration
   X = df.drop(["LONGITUDE", "LATITUDE", "SPACEID", "RELATIVEPOSITION"], axis = 1)
   X = pd.get_dummies(data=X, columns=["BUILDINGID", "FLOOR"])
   y = df[["LONGITUDE"]]
   return X, y   

# Latitude
def preprocess_alldata_latitude(df):
   """
   separates trainingData into independents and dependents
   will also be applied to validationData
   INPUT: Cleaned alldata dataframe
   OUTPUT: allData as independent and dependent variables
   """
   global X
   global y
   # split data into features and targets for another iteration
   X = df.drop(["LATITUDE", "SPACEID", "RELATIVEPOSITION"], axis = 1)
   X = pd.get_dummies(data=X, columns=["BUILDINGID", "FLOOR"])
   y = df[["LATITUDE"]]
   return X, y

# function for splitting
def split_data(preprocess_data_building):
    """
    splits into train and test
    """
    global X_train
    global X_test
    global y_train
    global y_test
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 42,
                                                        shuffle = True)
    # Show the results of the split
    print("Training set has {} samples." .format(X_train.shape[0]))
    print("Testing set has {} samples." .format(X_test.shape[0]))
    return X_train, X_test, y_train, y_test

# =============================================================================
# DATA CLEANING
# =============================================================================

# clean data
### Apply on trainingData
cleanedData = clean_data(trainingData.copy())
# look at data types
cleanedData.info()
# Bdg and floor should be categories

trainingData_c = cleanedData.copy()
trainingData_c["BUILDINGID"] = trainingData_c["BUILDINGID"].astype("category")
trainingData_c["FLOOR"] = trainingData_c["FLOOR"].astype("category")
trainingData_c["PHONEID"] = trainingData_c["PHONEID"].astype("category")

# check for rows where there is no WAP signal, ie rows that add to 0 across the 520 waps
trainingData_pure = trainingData_c.copy()
nilrows = trainingData_pure.iloc[:, 0:520].sum(axis=1)
nilrow_indices = nilrows[nilrows == 0].index
len(nilrow_indices) # total of 76 rows adding to 0
# drop the zero-columns
trainingData_pure = trainingData_pure.drop(trainingData_pure.index[nilrow_indices])
# make sure they're gone
checkrow = trainingData_pure.iloc[:, 0:520].sum(axis=1)
len(checkrow[checkrow == 0])
# make note of the empty rows and have a look at where they are at some point
#empty_rows = trainingData_c.iloc[nilrow_indices,:].copy()

### Apply on validationData
validationData_c = clean_data(validationData.copy())
validationData_c.info()
validationData_c["BUILDINGID"] = validationData_c["BUILDINGID"].astype("category")
validationData_c["FLOOR"] = validationData_c["FLOOR"].astype("category")
validationData_c["PHONEID"] = validationData_c["PHONEID"].astype("category")

# check for "zero-rows"
checkrows_V = validationData_c.iloc[:, 0:520].sum(axis=1)
len(checkrows_V[checkrows_V == 0])
# no zero-rows

### Apply on testData
testData_c = clean_data(testData.copy())
testData_c.info()
testData_c["BUILDINGID"] = testData_c["BUILDINGID"].astype("category")
testData_c["FLOOR"] = testData_c["FLOOR"].astype("category")
testData_c["PHONEID"] = testData_c["PHONEID"].astype("category")

# check for "zero-rows"
checkrows_T = testData_c.iloc[:, 0:520].sum(axis=1)
len(checkrows_T[checkrows_T == 0])
# no zero-rows

# =============================================================================
# EXPLORE CLEANED DATA
# =============================================================================

# Plot all the waps and explore the distribution of the RSSI values
## trainingData
histowap = pd.melt(trainingData_c, value_vars= col_names[0:520], var_name = "WAP", value_name = "RSSI")
histowap.iloc[:, 1].hist(bins=10)

# exclude the zero-signals to get a more descriptive plot
histowap_no_zeros = histowap.loc[histowap.RSSI != 0,:].copy()
histowap_no_zeros.hist(bins=10, histtype = "step")
#histowap_no_zeros.RSSI.mean()
#histowap_no_zeros.RSSI.median()
#histowap_no_zeros.RSSI.describe()
# right-skewed and an outlier cluster up top, check the 5% quantiles
histowap_no_zeros.quantile(0.025), histowap_no_zeros.quantile(0.975)
# the bottom 2.5% percentile is 10 and 97.5 percentile is 57

# check the distribution of means of each wap
md = trainingData_c.copy()
wapmeans = [md.iloc[:, i].mean() for i in range(520)]
df_wapmeans = pd.DataFrame(wapmeans.copy())
df_wapmeans.columns = ["WAPmeans"]
df_wapmeans["WAPmeans"].hist(histtype = "step")

# next check the means of the waps when they are activated to get an understanding
# of what an average signal could be 
br=[]
gr = []
mean = []

for i in range(520):
    gr = md.iloc[:, i]
    gr = gr.loc[gr != 0]
    mean = gr.mean()
    br.append(mean) 

wapmeans_nozeros = pd.DataFrame(br.copy())
wapmeans_nozeros.columns = ["WAPmeans"]
wapmeans_nozeros.WAPmeans.hist(histtype = "step")
wapmeans_nozeros.describe()
wapmeans_nozeros.quantile(0.025), wapmeans_nozeros.quantile(0.975) # quick n' dirty "confidence intervals"
# mean of means is around 22 with a sigma of 8

## validationData

histowap_val = pd.melt(validationData_c, value_vars= col_names[0:520], var_name = "WAP", value_name = "RSSI")
histowap_val.iloc[:, 1].hist(bins=10)

# exclude the zero-signals to get a more descriptive plot
histowap_v_no_zeros = histowap_val.loc[histowap_val.RSSI != 0,:].copy()
histowap_v_no_zeros.hist(bins=10, histtype = "step")
#histowap_v_no_zeros.RSSI.describe()
# right-skewed but no outlier cluster up top, check the 5% quantiles
histowap_v_no_zeros.quantile(0.025), histowap_v_no_zeros.quantile(0.975)
# the bottom 2.5% percentile is 10 and 97.5 percentile is 55

# check the distribution of means of each wap
mdv = validationData_c.copy()
wapmeans_v = [mdv.iloc[:, i].mean() for i in range(520)]
df_wapmeans_v = pd.DataFrame(wapmeans_v.copy())
df_wapmeans_v.columns = ["WAPmeans"]
df_wapmeans_v["WAPmeans"].hist(histtype = "step")

# next check the means of the waps when they are activated to get an understanding
# of what an average signal could be 
br_v=[]
gr_v = []
mean_v = []

for i in range(520):
    gr_v = mdv.iloc[:, i]
    gr_v = gr_v.loc[gr_v != 0]
    mean_v = gr_v.mean()
    br_v.append(mean_v) 

wapmeans_v_nozeros = pd.DataFrame(br_v.copy())
wapmeans_v_nozeros.columns = ["WAPmeans"]
wapmeans_v_nozeros.WAPmeans.hist(histtype = "step")
wapmeans_v_nozeros.describe()
wapmeans_v_nozeros.quantile(0.025), wapmeans_v_nozeros.quantile(0.975) # quick n' dirty "confidence intervals"
# mean of means is around 24 with a sigma of 6.4

# =============================================================================
# EXPLORE THE PHONES
# =============================================================================
# There are two confounders in the dataset one is the user the other is the phone
# because the users are of different length (etc.) and teh different phone models register signals differently
# we cannot affect them, but we can try to understand the phone and perhaps convert or normalize them

### Training Data
# print the phone ID's
print("Phone types in the Trainingset: {}".format(unique_phones))

# plot the frequency of the phonetypes
sns.countplot(x= "PHONEID", data = trainingData_pure)
sns.countplot(x= "PHONEID", data = validationData_c)
sns.countplot(x= "PHONEID", data = testData)


# filter all phones separately, melt, plot, exclude zero-values, plot again and then
# merge all into a dataframe that will hold all RSSI signals of all phonetypes of the training set
# Just do this manually

# 23
phone23 = trainingData_pure.loc[trainingData_pure.PHONEID == 23,:].copy()
phone23waps = pd.melt(phone23, value_vars= col_names[0:520], var_name = "WAP23", value_name = "RSSI")
phone23waps.hist(bins=100)
phone23waps.describe()
phone23wapsactive = phone23waps.loc[phone23waps.RSSI != 0,:].copy()
phone23wapsactive.hist(bins=100)
pr = pd.DataFrame()
pr["phone23"] = phone23wapsactive["RSSI"].copy()

#13
phone13 = trainingData_pure.loc[trainingData_pure.PHONEID == 13,:].copy()
phone13waps = pd.melt(phone13, value_vars= col_names[0:520], var_name = "WAP13", value_name = "RSSI")
phone13waps.hist(bins = 100)
phone13waps.describe()
phone13wapsactive = phone13waps.loc[phone13waps.RSSI != 0,:].copy()
phone13wapsactive.hist(bins=100)
pr["phone13"] = phone13wapsactive["RSSI"].copy()

#16
phone16 = trainingData_pure.loc[trainingData_pure.PHONEID == 16,:].copy()
phone16waps = pd.melt(phone16, value_vars= col_names[0:520], var_name = "WAP16", value_name = "RSSI")
#phone16waps.hist(bins=100)
phone16waps.describe()
phone16wapsactive = phone16waps.loc[phone16waps.RSSI != 0,:].copy()
phone16wapsactive.hist(bins=100)
pr["phone16"] = phone16wapsactive["RSSI"].copy()

#18
phone18 = trainingData_pure.loc[trainingData_pure.PHONEID == 18,:].copy()
phone18waps = pd.melt(phone18, value_vars= col_names[0:520], var_name = "WAP18", value_name = "RSSI")
#phone18waps.hist(bins=100)
phone18waps.describe()
phone18wapsactive = phone18waps.loc[phone18waps.RSSI != 0,:].copy()
phone18wapsactive.hist(bins=100)
pr["phone18"] = phone18wapsactive["RSSI"].copy()

# 3
phone03 = trainingData_pure.loc[trainingData_pure.PHONEID == 3,:].copy()
phone03waps = pd.melt(phone03, value_vars= col_names[0:520], var_name = "WAP03", value_name = "RSSI")
#phone03waps.hist(bins=100)
phone03waps.describe()
phone03wapsactive = phone03waps.loc[phone03waps.RSSI != 0,:].copy()
phone03wapsactive.hist(bins=100)
pr["phone03"] = phone03wapsactive["RSSI"].copy()

#19
phone19 = trainingData_pure.loc[trainingData_pure.PHONEID == 19,:].copy()
phone19waps = pd.melt(phone19, value_vars= col_names[0:520], var_name = "WAP19", value_name = "RSSI")
#phone19waps.hist(bins=100)
phone19waps.describe()
phone19wapsactive = phone19waps.loc[phone19waps.RSSI != 0,:].copy()
phone19wapsactive.hist(bins=100)
phone19wapsactive.describe()
pr["phone19"] = phone19wapsactive["RSSI"].copy()

#06
phone06 = trainingData_pure.loc[trainingData_pure.PHONEID == 6,:].copy()
phone06waps = pd.melt(phone06, value_vars= col_names[0:520], var_name = "WAP06", value_name = "RSSI")
#phone06waps.hist(bins=100)
phone06waps.describe()
phone06wapsactive = phone06waps.loc[phone06waps.RSSI != 0,:].copy()
phone06wapsactive.hist(bins=100)
pr["phone06"] = phone06wapsactive["RSSI"].copy()

# 1
phone01 = trainingData_pure.loc[trainingData_pure.PHONEID == 1,:].copy()
phone01waps = pd.melt(phone01, value_vars= col_names[0:520], var_name = "WAP01", value_name = "RSSI")
#phone01waps.hist(bins=100)
phone01waps.describe()
phone01wapsactive = phone01waps.loc[phone01waps.RSSI != 0,:].copy()
phone01wapsactive.hist(bins=100)
pr["phone01"] = phone01wapsactive["RSSI"].copy()

# 14
phone14 = trainingData_pure.loc[trainingData_pure.PHONEID == 14,:].copy()
phone14waps = pd.melt(phone14, value_vars= col_names[0:520], var_name = "WAP14", value_name = "RSSI")
#phone14waps.hist(bins=100)
phone14waps.describe()
phone14wapsactive = phone14waps.loc[phone14waps.RSSI != 0,:].copy()
phone14wapsactive.hist(bins=100)
pr["phone14"] = phone14wapsactive["RSSI"].copy()

# 8
phone08 = trainingData_pure.loc[trainingData_pure.PHONEID == 8,:].copy()
phone08waps = pd.melt(phone08, value_vars= col_names[0:520], var_name = "WAP08", value_name = "RSSI")
#phone08waps.hist(bins=100)
phone08waps.describe()
phone08wapsactive = phone08waps.loc[phone08waps.RSSI != 0,:].copy()
phone08wapsactive.hist(bins=100)
pr["phone08"] = phone08wapsactive["RSSI"].copy()

# 24
phone24 = trainingData_pure.loc[trainingData_pure.PHONEID == 24,:].copy()
phone24waps = pd.melt(phone24, value_vars= col_names[0:520], var_name = "WAP24", value_name = "RSSI")
#phone24waps.hist(bins=100)
phone24waps.describe()
phone24wapsactive = phone24waps.loc[phone24waps.RSSI != 0,:].copy()
phone24wapsactive.hist(bins=100)
pr["phone24"] = phone24wapsactive["RSSI"].copy()

# 17
phone17 = trainingData_pure.loc[trainingData_pure.PHONEID == 17,:].copy()
phone17waps = pd.melt(phone17, value_vars= col_names[0:520], var_name = "WAP17", value_name = "RSSI")
#phone17waps.hist(bins=100)
phone17waps.describe()
phone17wapsactive = phone17waps.loc[phone17waps.RSSI != 0,:].copy()
phone17wapsactive.hist(bins=100)
phone17wapsactive.describe()
pr["phone17"] = phone17wapsactive["RSSI"].copy()

# 7
phone07 = trainingData_pure.loc[trainingData_pure.PHONEID == 7,:].copy()
phone07waps = pd.melt(phone07, value_vars= col_names[0:520], var_name = "WAP07", value_name = "RSSI")
#phone07waps.hist(bins=100)
phone07waps.describe()
phone07wapsactive = phone07waps.loc[phone07waps.RSSI != 0,:].copy()
phone07wapsactive.hist(bins=100)
pr["phone07"] = phone07wapsactive["RSSI"].copy()

# 11
phone11 = trainingData_pure.loc[trainingData_pure.PHONEID == 11,:].copy()
phone11waps = pd.melt(phone11, value_vars= col_names[0:520], var_name = "WAP11", value_name = "RSSI")
#phone11waps.hist(bins=100)
phone11waps.describe()
phone11wapsactive = phone11waps.loc[phone11waps.RSSI != 0,:].copy()
phone11wapsactive.hist(bins=100)
phone11wapsactive.describe()
pr["phone11"] = phone11wapsactive["RSSI"].copy()

# 22
phone22 = trainingData_pure.loc[trainingData_pure.PHONEID == 22,:].copy()
phone22waps = pd.melt(phone22, value_vars= col_names[0:520], var_name = "WAP22", value_name = "RSSI")
#phone22waps.hist(bins=100)
phone22waps.describe()
phone22wapsactive = phone22waps.loc[phone22waps.RSSI != 0,:].copy()
phone22wapsactive.hist(bins=100)
pr["phone22"] = phone22wapsactive["RSSI"].copy()

# 10
phone10 = trainingData_pure.loc[trainingData_pure.PHONEID == 10,:].copy()
phone10waps = pd.melt(phone10, value_vars= col_names[0:520], var_name = "WAP10", value_name = "RSSI")
#phone10waps.hist(bins=100)
phone10waps.describe()
phone10wapsactive = phone10waps.loc[phone10waps.RSSI != 0,:].copy()
phone10wapsactive.hist(bins=100)
pr["phone10"] = phone10wapsactive["RSSI"].copy()

# define a function to plot a grid of histograms to make visual comparison easier, 
# use it on validation data phones too

def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=100,ax=ax)
    plt.show()

# plot the distributions
draw_histograms(pr, pr.columns, 4, 4)

### validationData
# dothe same as for training
print("Phone types in the Validationset: {}".format(unique_phones_val))

# 0
phone0 = validationData_c.loc[validationData_c.PHONEID == 0,:].copy()
phone0waps_v = pd.melt(phone0, value_vars= col_names[0:520], var_name = "WAP0", value_name = "RSSI")
phone0waps_v.hist(bins=100)
phone0waps_v.describe()
phone0wapsactive_v = phone0waps_v.loc[phone0waps_v.RSSI != 0,:].copy()
phone0wapsactive_v.hist(bins=100)
pr_v = pd.DataFrame()
pr_v["phone0"] = phone0wapsactive_v["RSSI"].copy()

# 13
phone13 = validationData_c.loc[validationData_c.PHONEID == 13,:].copy()
phone13waps_v = pd.melt(phone13, value_vars= col_names[0:520], var_name = "WAP13", value_name = "RSSI")
phone13waps_v.hist(bins=100)
phone13waps_v.describe()
phone13wapsactive_v = phone13waps_v.loc[phone13waps_v.RSSI != 0,:].copy()
phone13wapsactive_v.hist(bins=100)
pr_v["phone13"] = phone13wapsactive_v["RSSI"].copy()

# 2
phone2 = validationData_c.loc[validationData_c.PHONEID == 2,:].copy()
phone2waps_v = pd.melt(phone2, value_vars= col_names[0:520], var_name = "WAP2", value_name = "RSSI")
phone2waps_v.hist(bins=100)
phone2waps_v.describe()
phone2wapsactive_v = phone2waps_v.loc[phone2waps_v.RSSI != 0,:].copy()
phone2wapsactive_v.hist(bins=100)
pr_v["phone2"] = phone2wapsactive_v["RSSI"].copy()

# 12
phone12 = validationData_c.loc[validationData_c.PHONEID == 12,:].copy()
phone12waps_v = pd.melt(phone12, value_vars= col_names[0:520], var_name = "WAP12", value_name = "RSSI")
phone12waps_v.hist(bins=100)
phone12waps_v.describe()
phone12wapsactive_v = phone12waps_v.loc[phone12waps_v.RSSI != 0,:].copy()
phone12wapsactive_v.hist(bins=100)
pr_v["phone12"] = phone12wapsactive_v["RSSI"].copy()

# 20
phone20 = validationData_c.loc[validationData_c.PHONEID == 20,:].copy()
phone20waps_v = pd.melt(phone20, value_vars= col_names[0:520], var_name = "WAP20", value_name = "RSSI")
phone20waps_v.hist(bins=100)
phone20waps_v.describe()
phone20wapsactive_v = phone20waps_v.loc[phone13waps_v.RSSI != 0,:].copy()
phone20wapsactive_v.hist(bins=100)
pr_v["phone20"] = phone20wapsactive_v["RSSI"].copy()

# 21
phone21 = validationData_c.loc[validationData_c.PHONEID == 21,:].copy()
phone21waps_v = pd.melt(phone21, value_vars= col_names[0:520], var_name = "WAP21", value_name = "RSSI")
phone21waps_v.hist(bins=100)
phone21waps_v.describe()
phone21wapsactive_v = phone21waps_v.loc[phone21waps_v.RSSI != 0,:].copy()
phone21wapsactive_v.hist(bins=100)
pr_v["phone21"] = phone21wapsactive_v["RSSI"].copy()

# 4
phone4 = validationData_c.loc[validationData_c.PHONEID == 4,:].copy()
phone4waps_v = pd.melt(phone4, value_vars= col_names[0:520], var_name = "WAP4", value_name = "RSSI")
phone4waps_v.hist(bins=100)
phone4waps_v.describe()
phone4wapsactive_v = phone4waps_v.loc[phone4waps_v.RSSI != 0,:].copy()
phone4wapsactive_v.hist(bins=100)
pr_v["phone4"] = phone4wapsactive_v["RSSI"].copy()

# 9
phone9 = validationData_c.loc[validationData_c.PHONEID == 9,:].copy()
phone9waps_v = pd.melt(phone9, value_vars= col_names[0:520], var_name = "WAP9", value_name = "RSSI")
phone9waps_v.hist(bins=100)
phone9waps_v.describe()
phone9wapsactive_v = phone9waps_v.loc[phone9waps_v.RSSI != 0,:].copy()
phone9wapsactive_v.hist(bins=100)
pr_v["phone9"] = phone9wapsactive_v["RSSI"].copy()

# 15
phone15 = validationData_c.loc[validationData_c.PHONEID == 15,:].copy()
phone15waps_v = pd.melt(phone15, value_vars= col_names[0:520], var_name = "WAP15", value_name = "RSSI")
phone15waps_v.hist(bins=100)
phone15waps_v.describe()
phone15wapsactive_v = phone15waps_v.loc[phone15waps_v.RSSI != 0,:].copy()
phone15wapsactive_v.hist(bins=100)
pr_v["phone15"] = phone15wapsactive_v["RSSI"].copy()

# 5
phone5 = validationData_c.loc[validationData_c.PHONEID == 5,:].copy()
phone5waps_v = pd.melt(phone5, value_vars= col_names[0:520], var_name = "WAP5", value_name = "RSSI")
phone5waps_v.hist(bins=100)
phone5waps_v.describe()
phone5wapsactive_v = phone5waps_v.loc[phone5waps_v.RSSI != 0,:].copy()
phone5wapsactive_v.hist(bins=100)
pr_v["phone5"] = phone5wapsactive_v["RSSI"].copy()

# 14
phone14 = validationData_c.loc[validationData_c.PHONEID == 14,:].copy()
phone14waps_v = pd.melt(phone14, value_vars= col_names[0:520], var_name = "WAP14", value_name = "RSSI")
phone14waps_v.hist(bins=100)
phone14waps_v.describe()
phone14wapsactive_v = phone14waps_v.loc[phone14waps_v.RSSI != 0,:].copy()
phone14wapsactive_v.hist(bins=100)
pr_v["phone14"] = phone14wapsactive_v["RSSI"].copy()

# plot distributions in a grid

draw_histograms(pr_v, pr_v.columns, 4, 4)

# the distributions of the signal strenths of all the phone types are quite different
# let's try to reduce the noise caused by this inaccuracy/inconsistency in the data collection
# by regressing the signal strengths recorded for each phone toward the mean RSSI value for each WAP 

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

### Create dataframes for FE
# traininData
trainFeat_e = trainingData_pure.copy()

#validationData
valFeat_e = validationData_c.copy()

# testData
testFeat_e = testData_c.copy()

# create a feature "TOTALWAPS" that sums up all the RSSI readings for a row

# trainingData
trainFeat_e["TOTALWAP"] = trainFeat_e.iloc[:, 0:520].sum(axis = 1, skipna = True)
# explore TOTALWAP
sns.countplot(x= "TOTALWAP", data = trainFeat_e) # right skewed, but kinda bell shaped
trainFeat_e["TOTALWAP"].describe() # median not too far from the mean, Q3 far from max
sns.boxplot(x="TOTALWAP", data = trainFeat_e)

# validationData
valFeat_e["TOTALWAP"] = valFeat_e.iloc[:, 0:520].sum(axis = 1, skipna = True)
# explore TOTALWAP
sns.countplot(x= "TOTALWAP", data = valFeat_e) # right skewed again, but kinda bell shaped
valFeat_e["TOTALWAP"].describe() # max a lot lower than in trainset, but the other measurements are fairly similar
sns.boxplot(x="TOTALWAP", data = valFeat_e)

# testData
testFeat_e["TOTALWAP"] = testFeat_e.iloc[:, 0:520].sum(axis = 1, skipna = True)
# explore TOTALWAP
sns.countplot(x= "TOTALWAP", data = testFeat_e) # a lot less skewed than training and validation sets
testFeat_e["TOTALWAP"].describe() # mean close to median, sd relatively speaking a lot smaller than in the other sets, max closer to Q3
sns.boxplot(x="TOTALWAP", data = testFeat_e)

### AVERAGE RSSI PER WAP PER PHONE TYPE
# find out all the active waps for each row
# use booleans as a quick shortcut

# trainingData
trainWAPbool = trainFeat_e.iloc[:, 0:520].copy()
trainWAPbool = trainWAPbool.astype("bool")
trainWAPbool = trainWAPbool.astype("int")
trainWAPbool["ActiveWAPs"] = trainWAPbool.iloc[:, 0:520].sum(axis = 1, skipna = True)
trainFeat_e["ActiveWAPs"] = trainWAPbool["ActiveWAPs"].copy()

# validationData
valWAPbool = valFeat_e.iloc[:, 0:520].copy()
valWAPbool = valWAPbool.astype("bool")
valWAPbool = valWAPbool.astype("int")
valWAPbool["ActiveWAPs"] = valWAPbool.iloc[:, 0:520].sum(axis = 1, skipna = True)
valFeat_e["ActiveWAPs"] = valWAPbool["ActiveWAPs"].copy()

# testData
testWAPbool = testFeat_e.iloc[:, 0:520].copy()
testWAPbool = testWAPbool.astype("bool")
testWAPbool = testWAPbool.astype("int")
testWAPbool["ActiveWAPs"] = testWAPbool.iloc[:, 0:520].sum(axis = 1, skipna = True)
testFeat_e["ActiveWAPs"] = testWAPbool["ActiveWAPs"].copy()

# again, make new dataframes for the following step to make it easier to iterate with different datasets
trainFeat_em = trainFeat_e.copy() 
valFeat_em = valFeat_e.copy()
testFeat_em = testFeat_e.copy()

# get the avg rssi per active WAP for each row
# Training
trainFeat_em["SSpW"] = trainFeat_em.iloc[:, 529].div(trainFeat_em.iloc[:, 530], axis=0)

# extract values for each phone
ph = trainFeat_em.copy()
up = tuple(unique_phones.copy())
phonelist=[]

for i in up:
    pho = pd.DataFrame(ph.loc[ph.PHONEID == i,:].copy())
    phonelist.append(pho)

# Get the mean for each phone separately
avg = [phonelist[i]["SSpW"].mean() for i in range(len(phonelist))]
ph_conv_avg = pd.DataFrame(avg.copy())    
ph_conv_avg.columns = ["meanSSpw"]
ph_conv_avg["PHONEID"] = up
ph_conv_avg["meanSSpw"].hist(bins=16, histtype = "step")

# Get the mean of the sum of means of all phones
meanofmeans = ph_conv_avg.meanSSpw.mean()
ph_conv_avg["meanofmeans"] = meanofmeans

# extract a coefficient that equal to the mean rssi per wap of a certain phone divided by the mean across all phones
ph_conv_avg["conversion"] = ph_conv_avg.iloc[:, 0].div(ph_conv_avg.iloc[:, 2], axis = 0)

# Create a dictionary to to slot the 
# right conversion factor on each row
# key = phoneid, value = conversion factor
print("Keys for the factor dictionary for trainingData: {}".format(up))
print("Values for the factor dictionary for trainingData: {}".format(ph_conv_avg["conversion"]))
# fill in the values
factor_dictionary = {23 : 0.895204, 13 : 0.993342, 16 : 0.839080,
                   18: 0.813319, 3 : 1.143055, 19: 1.034812, 6: 1.095103,
                   1 : 0.921544, 14: 1.218480, 8: 0.998093, 24: 1.013398,
                   17: 1.157735, 7: 0.966760, 11: 1.199592, 22: 0.852915, 10: 0.857566} 

# Add a new column named 'Factor' 
trainFeat_em["Factor"] = trainFeat_em["PHONEID"].map(factor_dictionary) 

# again, create another dateframe to be used in other iterations
convertedwaps_df = trainFeat_em.copy()

# convert the wap values by dividing the wap features with the coefficient, 
# thereby regressing them toward the mean values of all phones
convertedwaps_df.iloc[:, 0:520] = convertedwaps_df.iloc[:, 0:520].div(convertedwaps_df["Factor"], axis = 0)

# drop now redundant columns
convertedwaps_df_c = convertedwaps_df.iloc[:, 0:529].copy()

# pickle 
joblib.dump(convertedwaps_df_c, "convertedwaps_df_c.pkl")


### repeat for validationData

valFeat_em["SSpW"] = valFeat_em.iloc[:, 529].div(valFeat_em.iloc[:, 530], axis=0)

# filter out each phone
ph_val = valFeat_em.copy()
up_val = tuple(unique_phones_val.copy())
phonelist_val=[]

for i in up_val:
    pho_val = pd.DataFrame(ph_val.loc[ph_val.PHONEID == i,:].copy())
    phonelist_val.append(pho_val)

avg_val = [phonelist_val[i]["SSpW"].mean() for i in range(len(phonelist_val))]
ph_conv_avg_val = pd.DataFrame(avg_val.copy())    
ph_conv_avg_val.columns = ["meanSSpw"]
ph_conv_avg_val["PHONEID"] = up_val
ph_conv_avg_val["meanSSpw"].hist(bins=16, histtype = "step")

# extract values
ph_conv_avg_val["meanofmeans"] = meanofmeans
ph_conv_avg_val["conversion"] = ph_conv_avg_val.iloc[:, 0].div(ph_conv_avg_val.iloc[:, 2], axis = 0)

# create dictionary
print("Keys for the factor dictionary for validationData: {}".format(up_val))
print("Values for the factor dictionary for validationData: {}".format(ph_conv_avg_val["conversion"]))

val_factor_dictionary = {0:1.175584 , 13 : 1.030939, 2:0.922549,
                   12:1.287922 , 20: 1.097151, 21: 0.944018, 4: 1.183175,
                   9 : 1.057589, 15: 1.247171 , 5: 1.202532, 14: 1.316814}

# convert
convertedwaps_val = valFeat_em.copy()

convertedwaps_val["Factor"] = convertedwaps_val["PHONEID"].map(val_factor_dictionary)
convertedwaps_val.iloc[:, 0:520] = convertedwaps_val.iloc[:, 0:520].div(convertedwaps_val["Factor"], axis = 0)

# drop redundant columns
convertedwaps_val_c = convertedwaps_val.iloc[:, 0:529].copy()

# pickle it
joblib.dump(convertedwaps_val_c, "convertedwaps_val_c.pkl")

### repeat for testData

testFeat_em["SSpW"] = testFeat_em.iloc[:, 529].div(testFeat_em.iloc[:, 530], axis=0)

# filter out each phone
ph_test = testFeat_em.copy()
up_test = tuple(unique_phones_test.copy())
phonelist_test=[]

for i in up_test:
    pho_test = pd.DataFrame(ph_test.loc[ph_test.PHONEID == i,:].copy())
    phonelist_test.append(pho_test)

# extract the values
avg_test = [phonelist_test[i]["SSpW"].mean() for i in range(len(phonelist_test))]
ph_conv_avg_test = pd.DataFrame(avg_test.copy())    
ph_conv_avg_test.columns = ["meanSSpw"]
ph_conv_avg_test["PHONEID"] = up_test
ph_conv_avg_test["meanSSpw"].hist(bins=16, histtype = "step")
ph_conv_avg_test["meanofmeans"] = meanofmeans
ph_conv_avg_test["conversion"] = ph_conv_avg_test.iloc[:, 0].div(ph_conv_avg_test.iloc[:, 2], axis = 0)

# Create the dictionary

print("Keys for the factor dictionary for testData: {}".format(up_test))
print("Values for the factor dictionary for validationData: {}".format(ph_conv_avg_test["conversion"]))

test_factor_dictionary = {15: 1.335101, 13: 1.272888 , 28:1.308584, 25:1.219032 , 26:1.400006, 27:1.393011, 29:1.176431 }

# convert
convertedwaps_test = testFeat_em.copy()

convertedwaps_test["Factor"] = convertedwaps_test["PHONEID"].map(test_factor_dictionary)
convertedwaps_test.iloc[:, 0:520] = convertedwaps_test.iloc[:, 0:520].div(convertedwaps_test["Factor"], axis = 0)

# drop redundant columns
convertedwaps_test_c = convertedwaps_test.iloc[:, 0:529].copy()

# pickle it
joblib.dump(convertedwaps_test_c, "convertedwaps_test_c.pkl")









