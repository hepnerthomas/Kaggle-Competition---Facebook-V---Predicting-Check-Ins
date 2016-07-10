### Thomas Hepner
### 6/18/2016
### Facebook: Identify the correct place for check-ins

# Import general libraries
import os
import time
import math
import random
import functools
import argparse
import uuid
import json

# Import statistical and machine learning libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA, FastICA
from bayes_opt import BayesianOptimization

# Set random seed
random.seed(0)

def MAP3(actuals, predictions):
    """ Mean Average Precision metric used for evaluation.
    """
    n = len(actuals)
    average_precisions = [0] * n
    actuals = actuals.astype(str)
    predictions = np.asarray(predictions)
    for i in range(0, n): 
        correct = actuals.iloc[i]
        precision_k = 0.0
        u = len(predictions[i])
        for j in range(0, u):                
            if(predictions[i][j] == correct):
                precision_k = 1.0 / (j + 1)
                average_precisions[i] = precision_k
                break
    map3 = np.asarray(average_precisions).mean()
    return map3
    
def load_split_data(grid_variable): 
    """ Load train_validation and validation data sets for testing and tuning different 
        machine learning models. 
    """
    
    # Set work directory
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//data//')
    
    # Load data
    train = pd.read_csv("train_modified.csv", sep = ",") 
    grid_variables = ['grid_cell_20x40', 'grid_cell_50x50', 'grid_cell_100x100', 'grid_cell_50x100', 'grid_cell_75x150', 'grid_cell_100x200']
    grid_variables.remove(grid_variable)
    train = train.drop(grid_variables, 1)
    train, test = train_test_split(train, test_size = 0.3, random_state = 0)

    # Return data
    return train, test        
    
def transform_data(train, test, weights):
    """ Transforms train and test data with feature weights.
    """   
    # Transform train data
    train['x'] *= weights[0]    
    train['y'] *= weights[1]       
    train['hour'] *= weights[2] 
    train['weekday'] *= weights[3]
    train['day_of_month'] *= weights[4]
    train['month'] *= weights[5]
    train['year'] *= weights[6]
    train['accuracy'] *= weights[7]
    train['x_d_y'] *= weights[8]
    train['x_t_y'] *= weights[9]

    # Transform test data
    test['x'] *= weights[0]    
    test['y'] *= weights[1]       
    test['hour'] *= weights[2] 
    test['weekday'] *= weights[3]
    test['day_of_month'] *= weights[4]
    test['month'] *= weights[5]
    test['year'] *= weights[6]    
    test['accuracy'] *= weights[7]
    test['x_d_y'] *= weights[8]
    test['x_t_y'] *= weights[9]
    
    # Return transformed data
    return train, test
    
def prepare_data(data       , w_x
                            , w_y
                            , w_hour
                            , w_weekday
                            , w_day_of_month
                            , w_month
                            , w_year
                            , w_accuracy
                            , w_x_d_y
                            , w_x_t_y):  #, weights):
    """ Transforms train and test data for specific grid cell with feature weights. 
        This function is used for Bayesian optimization.
    """
    # Transform data
    data['x'] *= w_x   
    data['y'] *= w_y       
    data['hour'] *= w_hour
    data['weekday'] *= w_weekday
    data['day_of_month'] *= w_day_of_month
    data['month'] *= w_month
    data['year'] *= w_year
    data['accuracy'] *= w_accuracy
    data['x_d_y'] *= w_x_d_y
    data['x_t_y'] *= w_x_t_y
    
    # Return prepared data
    return data

def process_grid_cell(train, test, grid_id, threshold, model, grid_variable
                            , w_x
                            , w_y
                            , w_hour
                            , w_weekday
                            , w_day_of_month
                            , w_month
                            , w_year
                            , w_accuracy
                            , w_x_d_y
                            , w_x_t_y): #, feature_weights):
    """ Creates model and generates predictions for row_ids in a particular grid cell.
    """
    start = time.time()
    # Filter data onto single grid cell
    train_cell = train[train[grid_variable] == grid_id]
    test_cell = test[test[grid_variable] == grid_id]
    test_ids = test_cell.index
    
    # Prepare data with feature weights
    train_cell = prepare_data(train_cell                            
                            , w_x = w_x
                            , w_y = w_y
                            , w_hour = w_hour
                            , w_weekday = w_weekday
                            , w_day_of_month = w_day_of_month
                            , w_month = w_month
                            , w_year = w_year
                            , w_accuracy = w_accuracy
                            , w_x_d_y = w_x_d_y
                            , w_x_t_y = w_x_t_y)  
                            
    test_cell = prepare_data(test_cell                            
                            , w_x = w_x
                            , w_y = w_y
                            , w_hour = w_hour
                            , w_weekday = w_weekday
                            , w_day_of_month = w_day_of_month
                            , w_month = w_month
                            , w_year = w_year
                            , w_accuracy = w_accuracy
                            , w_x_d_y = w_x_d_y
                            , w_x_t_y = w_x_t_y)
    
    # Remove place ids from train data with frequency below threshold
    place_counts = train_cell.place_id.value_counts()
    mask = place_counts[train_cell.place_id.values] >= threshold
    train_cell = train_cell.loc[mask.values]

    # Encode place id as labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_cell.place_id.values)
    X_train = train_cell.drop(['place_id', grid_variable], axis = 1).values
    X_test = test_cell.drop(['place_id', grid_variable], axis = 1).values
        
    # Build training classifier and predict
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)

    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3]).astype(str)   
    end = time.time()
    time_elapsed = (end - start)
    
    # Generate CV score
    map3 = MAP3(test_cell['place_id'], pred_labels)
    
    # Return data
    return pred_labels, test_ids, time_elapsed, map3

def process_grid(train, test, threshold, model, grid_ids, grid_variable #, weights): 
                            , w_x
                            , w_y
                            , w_hour
                            , w_weekday
                            , w_day_of_month
                            , w_month 
                            , w_year
                            , w_accuracy
                            , w_x_d_y
                            , w_x_t_y): 
    """ Iterates through all cells in grid, builds classifiers, and generates predictions. 
        Outputs submission.
    """            
    # Iterates through grid, building a predictive model for each grid cell
    #num_cells_preds = test[test[grid_variable].isin(grid_ids)].shape[0]
    preds = np.zeros((test.shape[0] + train.shape[0], 3), dtype=int)
    preds = preds.astype(str)
    execution_time = 0
    num_cells = len(grid_ids)
    total_score = []
    total_size = 0.0
    for i in range(num_cells): 
        # Applying classifier to one grid cell
        grid_id = grid_ids[i]
        pred_labels, test_ids, time_elapsed, map3 = process_grid_cell(train, test, grid_id, threshold, model, grid_variable
                            , w_x
                            , w_y
                            , w_hour
                            , w_weekday
                            , w_day_of_month
                            , w_month
                            , w_year
                            , w_accuracy
                            , w_x_d_y
                            , w_x_t_y) #, weights)

        # Increment variables
        weighted_map = map3 * 1.0 * len(test_ids)
        total_score.append(weighted_map)
        total_size += len(test_ids)

        # Print percentage of process completed
        execution_time = execution_time + time_elapsed
        print "    " + str(round(100.0 * i / num_cells, 2)) + '% complete: ' + str(round(execution_time, 3)) + ' seconds : MAP3 Score: ' + str(round(map3, 3))
   
    # Generate score
    #score = str(round(MAP3(test['place_id'][test[grid_variable].isin(grid_ids)], preds), 3)) 
    score = np.asarray(total_score).sum() / total_size
   
    # Print and Return predictions and MAP@3 score
    print "NN MAP3 Score: " + str(round(score,4))
    return score  
                             
#### Execute Code ####   
if __name__ == '__main__':
    """ Generates place_id label predictions for test data.
    """
    
    print "1. Loading data..."
    grid_variable = 'grid_cell_20x40'
    train, test = load_split_data(grid_variable) 
    grid_ids = random.sample(train[grid_variable].unique(), k = int(math.floor(len(train[grid_variable].unique()) * 0.005)))
       
    print "2. Transforming data..."
    #feature_weights = [200, 580, 5, 2, 1, 2, 1, 2, 100, 100]
    #feature_weights = [100, 72, 88, 39, 40, 56, 17, 81, 86, 91]
    # ttrain, ttest = transform_data(train, test, feature_weights)
       
    print "3. Building models and generating predictions..."
    model_nn = KNeighborsClassifier(n_neighbors = 25, n_jobs = -1, weights = 'distance', metric = 'manhattan')

    print "4. Build model with sample of data..."
    nn_threshold = 5
    
    score_nn = process_grid(train, test, threshold = nn_threshold, grid_ids = grid_ids #, weights = feature_weights)
                            , model = model_nn, grid_variable = grid_variable
                            , w_x = 200 # 100
                            , w_y = 580 # 72
                            , w_hour = 5 # 88
                            , w_weekday = 2 # 39
                            , w_day_of_month = 1 # 40
                            , w_month = 2 # 56
                            , w_year = 1 # 17
                            , w_accuracy = 2 #81
                            , w_x_d_y = 100 # 86
                            , w_x_t_y = 100 # 91
                            )
                              
    print "5. Execute Bayesian parameter optimization to select feature weights..."   
    
    ### Bayesian Optimization of Parameters ### 
    f = functools.partial(process_grid, train, test, threshold = nn_threshold, grid_ids = grid_ids, model = model_nn, grid_variable = grid_variable) #, weights = feature_weights)
    bo = BayesianOptimization(f=f,
                                  pbounds={
                                      'w_x': (80, 200), # (100, 1000)
                                      'w_y': (50, 150),  # (500, 2000)
                                      "w_hour": (50, 150), # (1, 10)
                                      "w_weekday": (20, 60), # (1, 10)
                                      "w_day_of_month": (20, 100), # (1,10)
                                      "w_month": (20, 80), # (1,10)
                                      "w_year": (0, 50), # (2,20)
                                      "w_accuracy": (1, 5), # (3,30)
                                      "w_x_d_y": (70, 200), # (3,30)
                                      "w_x_t_y": (70, 200) # (3,30)
                                      },
                                  verbose=True
                                  )
    
    bo.maximize(init_points = 2, n_iter = 1, acq = "ei", xi = 1.0) # 0.1
    for i in range(300):
        bo.maximize(n_iter = 1, acq = "ei", xi = 1.0) # exploration points
        bo.maximize(n_iter = 1, acq = "ei", xi = 1.0) # exploitation points
        
    print "6. Complete!!!" 
 
    
    

