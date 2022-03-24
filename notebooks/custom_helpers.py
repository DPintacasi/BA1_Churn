"""
add at top of notebook
    
    import customer_helpers as ch

access functions:
    
    ch.evaluate(y, y_pred, y_pred_prob)
    
"""

# ======= Packages =========

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,auc,roc_curve


# from sklearn.metrics import  confusion_matrix,classification_report
# ======= Functions ========


def evaluate(y, y_pred, y_pred_prob):
    
    """
    Evaluates model 
    
    Input
    ---------------------
    y: vector of true labels
    y_pred: vector of predicted label
    y_pred_prob: vector of probabilities [from .predict_proba() sklearn method]
    
    Output
    ---------------------
    Classification report on whole test set
    AUC on whole test set
    Precision@250
    AUC@250
    """
    
    print("-"*60)
    class_names = ['Did not Churn', 'Churn']
    
    print("Performance Over Whole Set")
    print("-"*60)
    print(classification_report(y, y_pred,target_names = class_names))
    
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    AUC = auc(fpr, tpr)
    print("AUC: %.2f \n" % AUC)
    print("-"*60)
    
    # extract only positive prob
    y_pred_prob_pos = [x[1] for x in y_pred_prob]
    
    # get top 250 "most probable" positive predictions
    df=pd.DataFrame({'y':y,'y_pred':y_pred,'y_pred_prob':y_pred_prob_pos})
    df = df.sort_values(by='y_pred_prob',ascending=False)
    y_pred_250 = df.head(250)
   
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_pred_250.y, y_pred_250.y_pred)
    fpr, tpr, thresholds = roc_curve(y_pred_250.y, y_pred_250.y_pred)
    AUC = auc(fpr, tpr)

    print("No. of TP (precision@250): %i" % matrix[-1,-1])
    print("AUC: %.3f" % AUC)
    print("-"*60)
    
    
def load_data(path):
    """ 
    Loads data given file path and applys the following:
        - Parses dates and converts them into # of years since 2018
            - birth date -> customer_age
            - customer_since_all -> customer_since_all_years
            - customer_since_bank -> customer_since_bank_years
        - Cast binary features as BOOL
        - Cast the following as CATEGORY 
            -'customer_occupation_code'
            -'customer_education'
            -'customer_children'
            -'customer_relationship'
                -NOTE: missing values are given new category
                        0 if numeric encoding
                        'unknown' if string encoding
        - Cast postal code as object  
    
    Input
    ---------------------
    File Path (str, path object)
    
    Output
    ---------------------
    Pandas DataFrame
    """
    
    
    print('-'*60)
    print('loading data...')
    data = pd.read_csv(path,parse_dates = [29,30,32])
    
    print('transforming dates...')
    
    data['customer_since_all_years'] = (2018-data.customer_since_all.dt.year)
    data['customer_since_bank_years'] = (2018-data.customer_since_bank.dt.year)
    data['customer_age'] = (2018-data.customer_birth_date.dt.year)

    print('cast types into bool, object, categorical...')

    # binary casting
    cols_binary = ['homebanking_active', 'has_homebanking',
           'has_insurance_21', 'has_insurance_23', 'has_life_insurance_fixed_cap',
           'has_life_insurance_decreasing_cap', 'has_fire_car_other_insurance',
           'has_personal_loan', 'has_mortgage_loan', 'has_current_account',
           'has_pension_saving', 'has_savings_account',
           'has_savings_account_starter', 'has_current_account_starter','customer_self_employed']
    data[cols_binary] = data[cols_binary].astype('bool')
    if 'target' in data.columns:
        data['target'] = data['target'].astype('bool')
        
    # casting gender as binary (current values are 0,1
    data['customer_gender'] = data['customer_gender']-1
    data['customer_gender'] = data['customer_gender'].astype('bool')
    
    # categorical casting  
    
    # numerical encoding
    cols_cat_num = ['customer_occupation_code', 'customer_education']
    data[cols_cat_num] = data[cols_cat_num].fillna(value = 0)
    data[cols_cat_num] = data[cols_cat_num].astype('category')

    # string encoding
    cols_cat_str = ['customer_children', 'customer_relationship']
    data[cols_cat_str] = data[cols_cat_str].fillna(value = 'unknown')
    data[cols_cat_str] = data[cols_cat_str].astype('category')

    # object casting
    cols_object = ['customer_postal_code']

    data[cols_object] = data[cols_object].astype('object')
    
    
    print('data loaded and casted')
    print('-'*60)

    
    return data
    
       
    
    

    
    