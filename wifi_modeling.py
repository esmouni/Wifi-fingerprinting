#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 19:48:17 2019

@author: eliassmouni
"""

"""
Run Clean Wifi EDA first.

"""
import os
#print("Current Working Directory " , os.getcwd())


#try:
# Change the current working Directory    
  #os.chdir(os.path.expanduser("~/Documents/Ubiqum/wifi task"))
  #print("Directory changed")
#except OSError:
  #print("Can't change the Current Working Directory")        

# basic
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

# =============================================================================
# LOAD PICKLED CONVERTED DATASETS
# =============================================================================

# trainingdata
convertedwaps_df_c = joblib.load('convertedwaps_df_c.pkl')

# validationdata
convertedwaps_val_c = joblib.load("convertedwaps_val_c.pkl")

# testdata
convertedwaps_test_c = joblib.load("convertedwaps_test_c.pkl")

# =============================================================================
# TRAINING MODELS
# =============================================================================

# define functions for preprocessing

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

# define function for mean confidence intervals assuming a T-distibution, 
    # with the degrees of freedom being relatively high it should not differ too much
    # from the z-scores of a normal distribution

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h

# define grid and kfold for gridsearchCV

# grid for randomforest
grid_params_RF = {'bootstrap': [True],
 'max_depth': [10, 30, 50, 100],
 "n_estimators": [10, 100]}

# stratified kfold for classification tasks due to class inbalances
skf = StratifiedKFold(n_splits = 5, random_state = 42) 

# "normal" kfold for regression tasks
kf = KFold(n_splits = 5, random_state = 42) 

# =============================================================================
# CLASSIFY BUILDING
# =============================================================================

"""
use a custom mix of training and validation data with converted wap readings 
to train and validate model
"""
drop_con = drop_cols(convertedwaps_df_c.copy())
drop_con_val = drop_cols(convertedwaps_val_c.copy())

X_Tr_bdg, y_Tr_bdg = preprocess_data_building(drop_con.copy())
X_trainConTrBdg, X_testConTrBdg, y_trainConTrBdg, y_testConTrBdg = split_data(preprocess_data_building)

X_Val_bdg, y_Val_bdg = preprocess_data_building(drop_con_val.copy())
X_trainConValBdg, X_testConValBdg, y_trainConValBdg, y_testConValBdg = split_data(preprocess_data_building)

X_ConTrainBdg = pd.concat([X_trainConTrBdg, X_trainConValBdg], ignore_index=True)
y_ConTrainBdg = pd.concat([y_trainConTrBdg, y_trainConValBdg], ignore_index=True)

X_ConTestBdg = pd.concat([X_testConTrBdg, X_testConValBdg], ignore_index=True)
y_ConTestBdg = pd.concat([y_testConTrBdg, y_testConValBdg], ignore_index=True)

# convert y to 1D array to speed up process
y_ConTrainBdg_A = np.array(y_ConTrainBdg.copy())
y_ConTrainBdg_A = y_ConTrainBdg_A.ravel()

# this takes a long time so I have removed it, you need to run it the first time though
# and after that you can just load the pickled model

# gridsearchcv to train model
#gs_RFstr_bdg = GridSearchCV(RandomForestClassifier(), grid_params_RF, cv = skf)
#gs_RFstr_bdg.fit(X_ConTrainBdg, y_ConTrainBdg_A)

# pickle the model first
#joblib.dump(gs_RFstr_bdg, "gs_bdg.pkl")

# load model
gs_RFstr_bdg = joblib.load("gs_bdg.pkl")

print("Best parameters: {}".format(gs_RFstr_bdg.best_params_))
print("Best cross-validation score: {}".format(gs_RFstr_bdg.best_score_))
print("Validation Set Score: {:.10f}".format(gs_RFstr_bdg.score(X_ConTestBdg, y_ConTestBdg)))
print(gs_RFstr_bdg.best_estimator_) # accuracy on validation 100%, this is the best one for building

# BOOTSTRAP VALIDATION

### BOOTSTRAP ####

bs_ValBdg = X_ConTestBdg.copy()
bs_ValBdg["BUILDINGID"] = y_ConTestBdg["BUILDINGID"].copy()

# loop for sampling
bs_bdg = []

for i in range(1000):
    np.random.seed = i
    bs_bdg.append((resample(bs_ValBdg)))

# check length    
print(len(bs_bdg))

# loop for splitting
X_list_Bdg=[]
y_list_Bdg=[]

for i in range(len(bs_bdg)):
    X_list_Bdg.append((bs_bdg[i].drop("BUILDINGID", axis=1)))
    y_list_Bdg.append((bs_bdg[i]["BUILDINGID"]))

# check lengths
len(X_list_Bdg)
len(y_list_Bdg)

# list comprehesion for results
results_bs_Bdg = [gs_RFstr_bdg.score(X_list_Bdg[i], y_list_Bdg[i]) for i in range(1000)]
len(results_bs_Bdg)

# confidence intervals (95)
results_bs_Bdg_df = pd.DataFrame(results_bs_Bdg.copy())
results_bs_Bdg_df.hist(bins = 1000, histtype = "step")
print(results_bs_Bdg_df.quantile(0.025)) 
print(results_bs_Bdg_df.quantile(0.975)) 
results_bs_Bdg_df.describe()

# this is a bit unnecessary given the above, but let's define a function for future use that will extract
# the confidence intervals using the sample standard deviation assuming students t-distribution 
# due to the many degrees of freedom, assuming a standard distribution might be more appropriate
# its not 100% accurate but that isn't crucial as the intervals are used to compare my models vis-a-vis each other


mean_confidence_interval(results_bs_Bdg_df, confidence = 0.95)

# =============================================================================
# FLOOR
# =============================================================================

"""
I will use cascading in classifying floor so that the predicted building will
be used as a feature. As the accuracy is 100%, I can just as well save myself
a few lines of code and use the observed building of the validation set instead
"""

### Preprocess data
X_Tr_fl, y_Tr_fl = preprocess_alldata_floor(drop_con.copy())
X_trainConTrFl, X_testConTrFl, y_trainConTrFl, y_testConTrFl = split_data(preprocess_alldata_floor)

X_Val_fl, y_Val_fl = preprocess_alldata_floor(drop_con_val.copy())
X_trainConValFl, X_testConValFl, y_trainConValFl, y_testConValFl = split_data(preprocess_alldata_floor)

X_ConTrainFl = pd.concat([X_trainConTrFl, X_trainConValFl], ignore_index=True)
y_ConTrainFl = pd.concat([y_trainConTrFl, y_trainConValFl], ignore_index=True)

X_ConTestFl = pd.concat([X_testConTrFl, X_testConValFl], ignore_index=True)
y_ConTestFl = pd.concat([y_testConTrFl, y_testConValFl], ignore_index=True)

# convert to 1D array
y_ConTrainFl_A = np.array(y_ConTrainFl.copy())
y_ConTrainFl_A = y_ConTrainFl_A.ravel()

# gridsearch CV
#gs_RFstr_floor = GridSearchCV(RandomForestClassifier(), grid_params_RF, cv = skf)
#gs_RFstr_floor.fit(X_ConTrainFl, y_ConTrainFl_A)

# pickle the model
#joblib.dump(gs_RFstr_floor, "gs_RFstr_floor.pkl")

# load model
gs_RFstr_floor = joblib.load("gs_RFstr_floor.pkl")

print("Best parameters: {}".format(gs_RFstr_floor.best_params_))
print("Best cross-validation score: {}".format(gs_RFstr_floor.best_score_))
print("Validation Set Score: {:.10f}".format(gs_RFstr_floor.score(X_ConTestFl, y_ConTestFl)))
print(gs_RFstr_floor.best_estimator_)
print(gs_RFstr_floor.cv_results_)
print(gs_RFstr_floor.cv_results_["std_test_score"])
print(np.average(gs_RFstr_floor.cv_results_["std_test_score"]))

# look closer at the errors
valpred_floor = pd.DataFrame(gs_RFstr_floor.predict(X_ConTestFl))
valpred_floor.columns = ["pFloor"]
print(classification_report(y_ConTestFl, valpred_floor))
y_ConTestFl.groupby('FLOOR').size()
valpred_floor.groupby('pFloor').size()
# errors on all floors, total of 6 misses (or 12 if you count both floors that were confused), 
# let's find the buildings where the errors were

ev_Floor = y_ConTestFl.copy()
ev_Floor["pFloor"] = valpred_floor["pFloor"]
ev_Floor["BUILDINGID"] = X_ConTestFl["BUILDINGID"].copy()

errors = [ev_Floor[(ev_Floor["FLOOR"] == i) & (ev_Floor["pFloor"] != i)] for i in range(5)]
errors

"""¨
bdg0 = 3 errors
bdg1 = 7 errors
bdg2 = 2 errors

"""

### BOOTSTRAP VALIDATION

bs_ValFl = X_ConTestFl.copy()
bs_ValFl["FLOOR"] = y_ConTestFl["FLOOR"].copy()

# loop for sampling
bs_fl = []

for i in range(1000):
    np.random.seed = i
    bs_fl.append((resample(bs_ValFl)))

# check length    
print(len(bs_fl))

# loop for splitting
X_list_Fl=[]
y_list_Fl=[]

for i in range(len(bs_fl)):
    X_list_Fl.append((bs_fl[i].drop("FLOOR", axis=1)))
    y_list_Fl.append((bs_fl[i]["FLOOR"]))

# check lengths
len(X_list_Fl)
len(y_list_Fl)

# list comprehesion for results
results_bs_Fl = [gs_RFstr_floor.score(X_list_Fl[i], y_list_Fl[i]) for i in range(1000)]
len(results_bs_Fl)

# confidence intervals (95)
results_bs_Fl_df = pd.DataFrame(results_bs_Fl.copy())
results_bs_Fl_df.hist(bins = 1000, histtype = "step")
print(results_bs_Fl_df.quantile(0.025)) 
print(results_bs_Fl_df.quantile(0.975)) 
results_bs_Fl_df.describe()


print("Confidence Intervals Floor: {}".format(mean_confidence_interval(results_bs_Fl_df, confidence = 0.95)))

# =============================================================================
# LONGITUDE
# =============================================================================

"""
I will train a randomforest regressor to predict longitude based on observed WAP 
values and predicted building and floor, I will first do one iteration with observed figures
to ascertain the predictive value of said features and then validate with cascaded features
"""

### Preprocess data
X_Tr_Lo, y_Tr_Lo = preprocess_alldata_longitude(drop_con.copy())
X_trainConTrLo, X_testConTrLo, y_trainConTrLo, y_testConTrLo = split_data(preprocess_alldata_longitude)

X_Val_Lo, y_Val_Lo = preprocess_alldata_longitude(drop_con_val.copy())
X_trainConValLo, X_testConValLo, y_trainConValLo, y_testConValLo = split_data(preprocess_alldata_longitude)

X_ConTrainLo = pd.concat([X_trainConTrLo, X_trainConValLo], ignore_index=True)
y_ConTrainLo = pd.concat([y_trainConTrLo, y_trainConValLo], ignore_index=True)

X_ConTestLo = pd.concat([X_testConTrLo, X_testConValLo], ignore_index=True)
y_ConTestLo = pd.concat([y_testConTrLo, y_testConValLo], ignore_index=True)

# make a 1D array
y_ConTrainLo_A = array(y_ConTrainLo.copy())
y_ConTrainLo_A = y_ConTrainLo_A.ravel()

# Gridsearch
#gs_RF_Long = GridSearchCV(RandomForestRegressor(), grid_params_RF, cv = kf)
#gs_RF_Long.fit(X_ConTrainLo, y_ConTrainLo_A)

#pickle model
#joblib.dump(gs_RF_Long, "gs_RF_Long.pkl")

# load model
gs_RF_Long = joblib.load("gs_RF_Long.pkl")

print("Best parameters: {}".format(gs_RF_Long.best_params_))
print("Best cross-validation score: {}".format(gs_RF_Long.best_score_))
print("Validation Set Score: {:.10f}".format(gs_RF_Long.score(X_ConTestLo, y_ConTestLo)))
print(gs_RF_Long.best_estimator_)

RFLo = gs_RF_Long.predict(X_ConTestLo)
print(sqrt(mean_squared_error(y_ConTestLo, RFLo)))
print(mean_absolute_error(y_ConTestLo,RFLo))

# check difference when using predicted floor
X_CascadeLo = X_ConTestLo.iloc[:, 0:523].copy()
X_CascadeLo["FLOOR"] = valpred_floor["pFloor"].copy()
X_CascadeLo = pd.get_dummies(data=X_CascadeLo, columns=["FLOOR"])

print("Cascade Validation Set Score: {:.10f}".format(gs_RF_Long.score(X_CascadeLo, y_ConTestLo)))
RFLoCas = gs_RF_Long.predict(X_CascadeLo)
print(sqrt(mean_squared_error(y_ConTestLo, RFLo)))
print(mean_absolute_error(y_ConTestLo,RFLo))
# results are exactly the same, so those errors on floor have no bearing on Longitude
# which makes intuitive sense given that z doesn't affect x and y


### BOOTSTRAP VALIDATION

# create appropriate dataframe
bs_ValLong = X_ConTestLo.copy()
bs_ValLong["LONGITUDE"] = y_ConTestLo["LONGITUDE"].copy()

bs_ValLong.head()

# loop for sampling
bs_Long = []

for i in range(1000):
    np.random.seed = i
    bs_Long.append((resample(bs_ValLong)))
    
print(len(bs_Long))

# loop for splitting
X_list_Long=[]
y_list_Long=[]

for i in range(len(bs_Long)):
    X_list_Long.append((bs_Long[i].drop("LONGITUDE", axis=1)))
    y_list_Long.append((bs_Long[i]["LONGITUDE"]))

print(len(X_list_Long))
print(len(y_list_Long))

# loop for predictions

Predlist_long = []

for i in range(len(X_list_Long)):
    Predlist_long.append((gs_RF_Long.predict(X_list_Long[i])))

print(len(Predlist_long))
# list comprehesion for results
MAE_Long = [mean_absolute_error(y_list_Long[i], Predlist_long[i]) for i in range(1000)]
len(MAE_Long)

# confidence intervals (95)
MAE_Long_df = pd.DataFrame(MAE_Long.copy())
MAE_Long_df.hist(bins = 1000, histtype = "step")
print(MAE_Long_df.quantile(0.025)) # 2.258831
print(MAE_Long_df.quantile(0.975)) # 2.476579
MAE_Long_df.mean() # 2.365599
mean_confidence_interval(MAE_Long_df, confidence = 0.95)


# =============================================================================
# LATITUDE
# =============================================================================

"""
I will train a randomforest regressor to predict longitude based on observed WAP 
values and predicted building and floor, I will first do one iteration with observed figures
to ascertain the predictive value of said features and then validate with cascaded features
"""

### Preprocess data
X_Tr_Lat, y_Tr_Lat = preprocess_alldata_latitude(drop_con.copy())
X_trainConTrLat, X_testConTrLat, y_trainConTrLat, y_testConTrLat = split_data(preprocess_alldata_latitude)

X_Val_Lat, y_Val_Lat = preprocess_alldata_latitude(drop_con_val.copy())
X_trainConValLat, X_testConValLat, y_trainConValLat, y_testConValLat = split_data(preprocess_alldata_latitude)

X_ConTrainLat = pd.concat([X_trainConTrLat, X_trainConValLat], ignore_index=True)
y_ConTrainLat = pd.concat([y_trainConTrLat, y_trainConValLat], ignore_index=True)

X_ConTestLat = pd.concat([X_testConTrLat, X_testConValLat], ignore_index=True)
y_ConTestLat = pd.concat([y_testConTrLat, y_testConValLat], ignore_index=True)

# make a 1D array
y_ConTrainLat_A = array(y_ConTrainLat.copy())
y_ConTrainLat_A = y_ConTrainLat_A.ravel()

# Gridsearch
#gs_RF_Lat = GridSearchCV(RandomForestRegressor(), grid_params_RF, cv = kf)
#gs_RF_Lat.fit(X_ConTrainLat, y_ConTrainLat_A)
# pickle model
#joblib.dump(gs_RF_Lat, "gs_RF_Lat.pkl")

# load model
gs_RF_Lat = joblib.load("gs_RF_Lat.pkl")

print("Best parameters: {}".format(gs_RF_Lat.best_params_))
print("Best cross-validation score: {}".format(gs_RF_Lat.best_score_))
print("Validation Set Score: {:.10f}".format(gs_RF_Lat.score(X_ConTestLat, y_ConTestLat)))
print(gs_RF_Lat.best_estimator_)
RFLat = gs_RF_Lat.predict(X_ConTestLat)
print(sqrt(mean_squared_error(y_ConTestLat, RFLat)))
print(mean_absolute_error(y_ConTestLat,RFLat))

# check difference when using predicted floor
X_CascadeLat = X_ConTestLat.iloc[:, 0:524].copy()
X_CascadeLat["FLOOR"] = valpred_floor["pFloor"].copy()
X_CascadeLat = pd.get_dummies(data=X_CascadeLat, columns=["FLOOR"])

print("Cascade Validation Set Score: {:.10f}".format(gs_RF_Lat.score(X_CascadeLat, y_ConTestLat)))
RFLatCas = gs_RF_Lat.predict(X_CascadeLat)
print(sqrt(mean_squared_error(y_ConTestLat, RFLat)))
print(mean_absolute_error(y_ConTestLat,RFLat))
# results are exactly the same, so those errors on floor have no bearing on Latitude
# which makes intuitive sense given that z doesn't affect x and y

# check difference when using predicted Longitude

X_CascadeLat["LONGITUDE"] = RFLoCas
print("Cascade Validation Set Score: {:.10f}".format(gs_RF_Lat.score(X_CascadeLat, y_ConTestLat)))
RFLatCas2 = gs_RF_Lat.predict(X_CascadeLat)
print(sqrt(mean_squared_error(y_ConTestLat, RFLatCas2)))
print(mean_absolute_error(y_ConTestLat,RFLatCas2))

### BOOTSTRAP VALIDATION

# create appropriate dataframe
bs_ValLat = X_CascadeLat.copy()
bs_ValLat["LATITUDE"] = y_ConTestLat["LATITUDE"].copy()

bs_ValLat.head()

# loop for sampling
bs_Lat = []

for i in range(1000):
    np.random.seed = i
    bs_Lat.append((resample(bs_ValLat)))
    
print(len(bs_Lat))

# loop for splitting
X_list_Lat=[]
y_list_Lat=[]

for i in range(len(bs_Lat)):
    X_list_Lat.append((bs_Lat[i].drop("LATITUDE", axis=1)))
    y_list_Lat.append((bs_Lat[i]["LATITUDE"]))

print(len(X_list_Lat))
print(len(y_list_Lat))

# loop for predictions

Predlist_lat = []

for i in range(len(X_list_Lat)):
    Predlist_lat.append((gs_RF_Lat.predict(X_list_Lat[i])))

print(len(Predlist_lat))

# list comprehesion for results
MAE_Lat = [mean_absolute_error(y_list_Lat[i], Predlist_lat[i]) for i in range(1000)]
len(MAE_Lat)

# confidence intervals (95)
MAE_Lat_df = pd.DataFrame(MAE_Lat.copy())
MAE_Lat_df.hist(bins = 1000, histtype = "step")
print(MAE_Lat_df.quantile(0.025)) 
print(MAE_Lat_df.quantile(0.975)) 
MAE_Lat_df.mean() 
mean_confidence_interval(MAE_Lat_df, confidence = 0.95)


