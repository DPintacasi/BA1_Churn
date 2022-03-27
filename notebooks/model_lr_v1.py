#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline

import custom_helpers as ch

np.random.seed(0)


# In[2]:


df = ch.load_data('../data/train_month_3_with_target.csv')
# print(df.info())


# In[3]:


#non sample-dependent transformations
def sample_agnostic_transformation(data):

    selected_col = [
                'homebanking_active'
                    # ,'has_homebanking'
                ,'bal_mortgage_loan'
                ,'has_life_insurance_decreasing_cap'
                    # # ,'has_mortgage_loan'
                ,'has_current_account'
                    # ,'cap_life_insurance_decreasing_cap'
                ,'bal_savings_account'
                ,'bal_current_account'
                ,'has_personal_loan'
                    # ,'bal_personal_loan'
                ,'customer_since_all_years'
                    # ,'customer_since_bank_years'
                ,'customer_age'
                ,'customer_children'
                ,'customer_education'
                # ,'has_savings_account'
                # ,'visits_distinct_so'
         ]
    
    if 'target' in data.columns:
        y = data.target
        X = data.drop(columns = ['target'])
        X = X[selected_col]
    else:
        X = data[selected_col]
        y = 0
        
    return X, y

X, y = sample_agnostic_transformation(df)


# In[4]:


# sample dependent column specific preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

num_col = X_train.select_dtypes(include = 'number', exclude = 'bool').columns
cat_col = X_train.select_dtypes(include = 'category').columns
bool_col = X_train.select_dtypes(include = 'bool').columns
date_col = X_train.select_dtypes(include = 'datetime64').columns
obj_col = X_train.select_dtypes(include = 'object').columns


# In[5]:


from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer

numeric_transformer = Pipeline(steps = [
    ('impute',SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scale', StandardScaler())
])

categorical_transformer = OneHotEncoder(drop = 'first',handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ('drop_ID','drop',obj_col),
        ('drop_dates','drop',date_col),
        ('cat',categorical_transformer,cat_col),
        ('num',numeric_transformer,num_col)
    ],
    remainder = "passthrough"
)

f = preprocessor.fit_transform(X_train)
f = pd.DataFrame(f)


# In[6]:


from sklearn.metrics import RocCurveDisplay

models = [LogisticRegression(class_weight = 'balanced')
          ,LogisticRegressionCV(cv = 5, random_state=0, class_weight = 'balanced' )
          ,LogisticRegressionCV(cv = 10, random_state=0, class_weight = 'balanced')
         ]

plt.figure()
for model in models:
    pipe = Pipeline(
    steps=[("preprocessor", preprocessor),("classifier", model)]
    )

    # train 
    clf = pipe.fit(X_train,y_train)
    
    # make prediction on test
    y_pred_test = clf.predict(X_test)
    y_pred_test_probs = clf.predict_proba(X_test)
    
    print(model)
    ch.evaluate(y_test, y_pred_test, y_pred_test_probs)
    print('\n')


# In[7]:


lr = LogisticRegression(max_iter=10000, tol=0.1, class_weight = 'balanced')

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("logistic", lr)]
)

param_grid = {
    # "logistic__C": np.logspace(-4, 4, 4)
    "logistic__C": np.logspace(-10,3,10)
}

print(param_grid)

gridscorer = ch.gridscorer() # customer scorer (precision@250)

search = GridSearchCV(pipe, param_grid, scoring = gridscorer, n_jobs=-2)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

lr = LogisticRegression(max_iter=10000
                        , tol=0.1
                        , class_weight = 'balanced'
                        , C = search.best_params_['logistic__C'])

#pipeline
pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("logistic", lr)]
)

clf = pipe.fit(X_train,y_train)

# make prediction on test
y_pred_test = clf.predict(X_test)
y_pred_test_probs = clf.predict_proba(X_test)

ch.evaluate(y_test, y_pred_test, y_pred_test_probs)

# run on submission data
# data_sub = ch.load_data('../data/test_month_3.csv')
# X_sub, y_sub = sample_agnostic_transformation(data_sub)

# # make prediction on test
# y_pred_sub = clf.predict(X_sub)
# y_pred_test_sub = clf.predict_proba(X_sub)
# y_pred_test_sub_pos = [x[1] for x in y_pred_test_sub]

# df = pd.DataFrame({'ID': data_sub.client_id,'PROB':y_pred_test_sub_pos})
# today = dt.datetime.today()
# df.to_csv(f'../output/lr_{today.month}{today.day}2.csv', index = False)
# In[ ]:




