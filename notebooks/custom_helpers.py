# """
# add at top of notebook
    
#     import customer_helpers as ch

# access functions:
    
#     ch.evaluate(y, y_pred, y_pred_prob)
    
# """

# ======= Packages =========

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix, classification_report,auc,roc_curve


# from sklearn.metrics import  confusion_matrix,classification_report
# ======= Functions ========


def evaluate(y, y_pred, y_pred_prob):
    
#     """
#     Evaluates model 
    
#     Input
#     ---------------------
#     y: vector of true labels
#     y_pred: vector of predicted label
#     y_pred_prob: vector of probabilities [from .predict_proba() sklearn method]
    
#     Output
#     ---------------------
#     Classification report on whole test set
#     AUC on whole test set
#     Precision@250
#     AUC@250
#     """
    
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
    
    
    

    
    