import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder


def load_data(path):
    """ 
    Loads data given file path and applies the following:
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
    
    
    data = pd.read_csv(path,index_col='client_id', parse_dates = [29,30,32])
    
    #Update Joris: I just removed the dates instead of making additional 
    #columns
    data['customer_since_all'] = (2018-data.customer_since_all.dt.year)
    data['customer_since_bank'] = (2018-data.customer_since_bank.dt.year)
    data['customer_birth_date'] = (2018-data.customer_birth_date.dt.year)

    # binary casting
    cols_binary = ['homebanking_active', 
                   'has_homebanking',
                   'has_insurance_21', 
                   'has_insurance_23', 
                   'has_life_insurance_fixed_cap',
                   'has_life_insurance_decreasing_cap', 
                   'has_fire_car_other_insurance',
                   'has_personal_loan', 
                   'has_mortgage_loan', 
                   'has_current_account',
                   'has_pension_saving', 
                   'has_savings_account',
                   'has_savings_account_starter', 
                   'has_current_account_starter',
                   'customer_self_employed']
    
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
    #change by Joris: Postal codes based on regions (only keep the thousands)
    data['customer_postal_code'] = (data['customer_postal_code']/1000).apply(np.floor)
    data['customer_postal_code'] = data['customer_postal_code'].astype('category')

    # string encoding
    cols_cat_str = ['customer_children', 'customer_relationship']
    data[cols_cat_str] = data[cols_cat_str].fillna(value = 'unknown')
    data[cols_cat_str] = data[cols_cat_str].astype('category')

    # object casting
    #cols_object = ['customer_postal_code']
    #data[cols_object] = data[cols_object].astype('object')

    return data 


def Onehotencoding(data):
  
    """ 
    takes a dataframe and applies One-hot encoding
    
    Input
    ---------------------
    Pandas DataFrame
    
    Output
    ---------------------
    Pandas DataFrame
    
    """
    scaler = OneHotEncoder(sparse=False)

    scale_category = data.select_dtypes(include="category").columns
    oneHotResults = scaler.fit_transform(data[scale_category])
    
    # Get all OneHotEncoder headers
    columns = scaler.categories_
    OneHotNames = []
    for i in range(len(columns)):
        OneHotNames = np.concatenate((OneHotNames, columns[i]))
    
    # Set client index for new dummies for merging
    index = data.index
    oneHotdf = pd.DataFrame(oneHotResults, columns=OneHotNames)
    oneHotdf = oneHotdf.set_index(index)
    
    # Remove old category features
    for i in range(len(scale_category)):
        data = data.drop(columns = [scale_category[i]])

    # Add new category features
    data = data.join(oneHotdf)
    
    return data


def missingtomean(data):
  
    """ 
    transforms missing values to the mean, also returns the mean for later use on the test set.
    
    Input
    ---------------------
    Pandas DataFrame Column
    
    Output
    ---------------------
    Pandas DataFrame Column
    Mean
    
    """
    
    mean = data.mean()
    data = data.fillna(mean)
    
    return data, mean