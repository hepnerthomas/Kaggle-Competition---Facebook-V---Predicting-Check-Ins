### Thomas Hepner
### 6/18/2016
### Facebook: Identify the correct place for check-ins

# Import libraries
import os
import time
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
import operator
    
def load_predictions(): 
    """ Load prediction data. 
    """
    ### Set work directory
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//submissions//')
    
    ### Load data
    
    ### XGB data
    preds1 = pd.read_csv("submission_XGB_thresh10_50_0.25_50x100.csv", sep = ",") 
    preds2 = pd.read_csv("submission_XGB_thresh10_50_0.25_75x150.csv", sep = ",")      
    preds3 = pd.read_csv("submission_XGB_thresh10_50_0.25_100x100.csv", sep = ",")   

    ### RF data
    preds4 = pd.read_csv("submission_RandomForest_thresh10_n50_0.56148.csv", sep = ",")     
    #preds5 = pd.read_csv("submission_RandomForest_thresh10_100_100x100.csv", sep = ",") 
    preds5 = pd.read_csv("submission_RandomForest_thresh10_100_50x50.csv", sep = ",") 
    
    ### NN data
    preds6 = pd.read_csv("submission_nn_thresh5_neighbors25_20x40_tuned_weights.csv", sep = ",")    
    preds7 = pd.read_csv("submission_nn_thresh5_neighbors25_50x50_tuned_weights_v2.csv", sep = ",")  
    #preds6 = pd.read_csv("submission_nn_thresh5_neighbors25_50x100_tuned_weights_v2.csv", sep = ",")  
    
    # Return data
    return preds1, preds2, preds3, preds4, preds5, preds6, preds7
    
def aggregate_predictions():
    """ Combined prediction data into single dataframe.
    """
    combined = pd.DataFrame()
    combined['preds1'] = preds1['place_id']
    combined['preds2'] = preds2['place_id']
    combined['preds3'] = preds3['place_id']
    combined['preds4'] = preds4['place_id']    
    combined['preds5'] = preds5['place_id']   
    combined['preds6'] = preds6['place_id']  
    combined['preds7'] = preds7['place_id']
    #combined['preds8'] = preds8['place_id']    
    #combined['preds9'] = preds9['place_id']    
    return combined
    
def combine_classifiers():
    """ Takes dataframe of predictions and outputs 
    """    
    start = time.time()
    predictions = []
    weights = [1.0, 0.5, 1.0/3.0]
    # iterate by row through dataframe
    for i in range(preds1.shape[0]):
#        place_ids1 = data['rf_preds'][i].split()
#        place_ids2 = data['xgb_preds'][i].split()
        place_ids1 = preds1['place_id'][i].split()
        place_ids2 = preds2['place_id'][i].split()        
        place_ids3 = preds3['place_id'][i].split()    
        place_ids4 = preds4['place_id'][i].split()          
        place_ids5 = preds5['place_id'][i].split()       
        place_ids6 = preds6['place_id'][i].split()     
        place_ids7 = preds7['place_id'][i].split()         
        #place_ids8 = preds8['place_id'][i].split()         
        #place_ids9 = preds9['place_id'][i].split()         
        dict = {}
        # iterate through ranked predictions for each model
        for j in range(0,3):
            ### Place Id 1
            if(dict.get(place_ids1[j]) is None):
                dict[place_ids1[j]] = weights[j]
            else: 
                dict[place_ids1[j]] += weights[j]
                
            ### Place Id 2
            if(dict.get(place_ids2[j]) is None):
                dict[place_ids2[j]] = weights[j] 
            else:
                dict[place_ids2[j]] += weights[j]   
                
            ### Place Id 3
            if(dict.get(place_ids3[j]) is None):
                dict[place_ids3[j]] = weights[j] 
            else:
                dict[place_ids3[j]] += weights[j] 
            
            ### Place Id 4
            if(dict.get(place_ids4[j]) is None):
                dict[place_ids4[j]] = weights[j] 
            else:
                dict[place_ids4[j]] += weights[j] 
                
            ### Place Id 5
            if(dict.get(place_ids5[j]) is None):
                dict[place_ids5[j]] = weights[j] 
            else:
                dict[place_ids5[j]] += weights[j]           
                
            ### Place Id 6
            if(dict.get(place_ids6[j]) is None):
                dict[place_ids6[j]] = weights[j] 
            else:
                dict[place_ids6[j]] += weights[j]   
                
            ### Place Id 7
            if(dict.get(place_ids7[j]) is None):
                dict[place_ids7[j]] = weights[j] 
            else:
                dict[place_ids7[j]] += weights[j]    
                
            ### Place Id 8
#            if(dict.get(place_ids8[j]) is None):
#                dict[place_ids8[j]] = weights[j] 
#            else:
#                dict[place_ids8[j]] += weights[j]                 
                
            ### Place Id 9
#            if(dict.get(place_ids9[j]) is None):
#                dict[place_ids9[j]] = weights[j] 
#            else:
#                dict[place_ids9[j]] += weights[j]  

                
        lst = sorted(dict.items(), key = operator.itemgetter(1), reverse = True)[0:3]
        # add sorted list to final prediction list
        temp = str(lst[0][0]) + ' ' + str(lst[1][0]) + ' ' + str(lst[2][0])
        predictions.append(temp)
    
    end = time.time()
    time_elapsed = (end - start) 
    
    # Return final predictions
    return predictions, time_elapsed
    
#### Execute Code ####   
if __name__ == '__main__':
    """ Generates place_id label predictions for test data.
    """
    
    print "1. Loading predictions..."
    preds1, preds2, preds3, preds4, preds5, preds6, preds7 = load_predictions() 
    
    print "2. Aggregating predictions..."
    combined = aggregate_predictions()
    
    print "3. Creating final predictions..."
    predictions, time_elapsed = combine_classifiers()    
    
    print "4. Writing to csv..."
    final_df = pd.DataFrame()
    final_df['row_id'] = preds1['row_id']
    final_df['place_id'] = predictions
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//final submissions//')  
    final_df.to_csv("XGB_3models_NN_2models_RF_2models.csv", index = False, header = True)
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//submissions//')  
    
    print ""
    print "Complete!!!"    
    
