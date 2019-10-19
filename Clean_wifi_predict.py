#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 20:58:40 2019

@author: eliassmouni
"""

""" 
run clean_wifi_EDA and wifi_modeling first

"""

# =============================================================================
# LOADING PACKAGES
# =============================================================================

import os
print("Current Working Directory " , os.getcwd())


#try:
#Change the current working Directory    
  #os.chdir(os.path.expanduser("~/Documents/Ubiqum/wifi task"))
  #print("Directory changed")
#except OSError:
 # print("Can't change the Current Working Directory")        

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
# LOAD MODELS AND DATA
# =============================================================================

# testdata
convertedwaps_test_c = joblib.load("convertedwaps_test_c.pkl")

# model for Building
gs_RFstr_bdg = joblib.load("gs_bdg.pkl")

# model for Floor
gs_RFstr_floor = joblib.load("gs_RFstr_floor.pkl")

# model for Longitude
gs_RF_Long = joblib.load("gs_RF_Long.pkl")

# model for latitude
gs_RF_Lat = joblib.load("gs_RF_Lat.pkl")

# =============================================================================
# DEFINE PREPROCESSING
# =============================================================================

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

# =============================================================================
# BUILDING
# =============================================================================

d_test = drop_cols(convertedwaps_test_c.copy())

X_bdg, y_bdg = preprocess_data_building(d_test.copy())

d_test["BUILDINGID"] = gs_RFstr_bdg.predict(X_bdg)


# =============================================================================
# FLOOR
# =============================================================================

X_floor, y_floor = preprocess_alldata_floor(d_test.copy())

d_test["FLOOR"] = gs_RFstr_floor.predict(X_floor)

# =============================================================================
# LONGITUDE
# =============================================================================

X_Lo, y_Lo= preprocess_alldata_longitude(d_test.copy())

d_test["LONGITUDE"] = gs_RF_Long.predict(X_Lo)

# =============================================================================
# LATITUDE
# =============================================================================

X_Lat, y_Lat = preprocess_alldata_latitude(d_test.copy())
d_test["LATITUDE"] = gs_RF_Lat.predict(X_Lat)

# =============================================================================
# SUBMISSION
# =============================================================================

submission_df = d_test[["LATITUDE", "LONGITUDE", "FLOOR"]].copy()
submission_df.to_csv("wf_submission.csv", index = False)




