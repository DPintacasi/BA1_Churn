# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:44:22 2022

@author: joris
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from sklearn import pipeline as pl
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

import Helpers as hlp

churn_now = hlp.load_data('train_month_3_with_target.csv')
churn_1mback = hlp.load_data('train_month_2.csv')
churn_2mback = hlp.load_data('train_month_1.csv')

datatypes = churn_now.dtypes

to_scale = churn_now.select_dtypes(include="category").columns



print(churn_now[to_scale].dtypes)