### Thomas Hepner
### 6/18/2016
### Facebook: Identify the correct place for check-ins

# Import general libraries
import os
import time
import math
import random

# Import statistical and machine learning libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

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

def process_grid_cell(train, test, grid_id, threshold, model, grid_variable):
    """ Creates model and generates predictions for row_ids in a particular grid cell.
    """
    start = time.time()
    # Filter data onto single grid cell
    train_cell = train[train[grid_variable] == grid_id]
    test_cell = test[test[grid_variable] == grid_id]
    test_ids = test_cell.index

    # Remove place ids from train data with frequency below threshold
    place_counts = train_cell.place_id.value_counts()
    mask = place_counts[train_cell.place_id.values] >= threshold
    train_cell = train_cell.loc[mask.values]

    # Encode place id as labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_cell.place_id.values)
    X_train = train_cell.drop(['place_id', grid_variable], axis = 1).values
    X_test = test_cell.drop(['place_id', grid_variable], axis = 1).values
        
    # NN as features
    model_nn = KNeighborsClassifier(n_neighbors = 31, n_jobs = -1, weights = 'distance', metric = 'manhattan')
    model_nn.fit(X_train, y_train)
    train_neighbors = pd.DataFrame(model_nn.kneighbors(X_train, n_neighbors = 31, return_distance = True)[0])
    test_neighbors = pd.DataFrame(model_nn.kneighbors(X_test, n_neighbors = 31, return_distance = True)[0])
    train_nn_cols = train_neighbors.columns
    test_nn_cols = test_neighbors.columns    
    
    train_cell[train_nn_cols] = train_neighbors
    train_cell[train_nn_cols] = train_neighbors.values
    
    test_cell[test_nn_cols] = test_neighbors
    test_cell[test_nn_cols] = test_neighbors.values 
   
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

def process_grid(train, test, threshold, model, grid_ids, grid_variable): 
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
        pred_labels, test_ids, time_elapsed, map3 = process_grid_cell(train, test, grid_id, threshold, model, grid_variable)

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
   
    # Return predictions and MAP@3 score
    print "MAP3 Score: " + str(round(score,3))
    return score, execution_time    
                             
#### Execute Code ####   
if __name__ == '__main__':
    """ Generates place_id label predictions for test data.
    """
    
    print "1. Loading data..."
    grid_variable = 'grid_cell_100x200'
    train, test = load_split_data(grid_variable) 
    grid_ids = random.sample(train[grid_variable].unique(), k = int(math.floor(len(train[grid_variable].unique()) * 0.005)))
    
    feature_weights = [200, 580, 5, 2, 1, 2, 1, 2, 100, 100]          
    ttrain, ttest = transform_data(train, test, weights = feature_weights)          
              
    print "2. Building models and generating predictions..."
    model_xgb = xgb.XGBClassifier(objective = "multi:softprob", max_depth = 3, n_estimators = 50, learning_rate = 0.25, gamma = 0.4, min_child_weight = 2, reg_alpha = 1, nthread = -1, seed = 0) 
    model_rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = 0)

    print "3. Build model with sample of data..."
    rf_threshold = 10
    xgb_threshold = 10
    
    score_xgb, time_xgb = process_grid(ttrain, ttest, threshold = xgb_threshold, grid_ids = grid_ids, model = model_xgb, grid_variable = grid_variable)   
    score_rf, time_rf = process_grid(train, test, threshold = rf_threshold, grid_ids = grid_ids, model = model_rf, grid_variable = grid_variable)

    print "XGB MAP3 Score: " + str(round(score_xgb,3)) + ' , Time Elapsed: ' + str(round(time_xgb, 1))
    print "Random Forest MAP3 Score: " + str(round(score_rf,3)) + ' , Time Elapsed: ' + str(round(time_rf, 1))
    
    print "4. Complete!!!"    
    
    ### TESTING!!! ###
#    grid_id = 2 # 315, 2, 680, 487
#    xgb_threshold = 40
#    model_xgb = xgb.XGBClassifier(objective = "multi:softprob", max_depth = 3, n_estimators = 100, learning_rate = 0.1, nthread = -1, seed = 0) 
#    pred_labels_xgb, test_ids_xgb, time_elapsed_xgb, xgb_score = process_grid_cell(train, test, grid_id = grid_id, threshold = xgb_threshold, model = model_xgb, grid_variable = grid_variable)       
#    print "XGB Score: " + str(round(xgb_score,3)) + " , Time Elapsed: " + str(round(time_elapsed_xgb,1))

#    # Find most important features
#    imp_features_rf = model_xgb.feature_importances_
#    n = len(imp_features_rf)
#    
#    # make importances relative to max importance
#    imp_features_rf = 100.0 * (imp_features_rf / imp_features_rf.max())
#    sorted_idx_rf = np.argsort(imp_features_rf)
#    pos_rf = np.arange(sorted_idx_rf.shape[0]) + .5
#    
#    # RF importance plot
#    plt.subplot(1, 2, 2)
#    plt.barh(pos_rf, imp_features_rf[sorted_idx_rf], align='center')
#    plt.yticks(pos_rf, train.drop(['place_id', grid_variable], 1).columns[sorted_idx_rf])
#    plt.xlabel('Relative Importance')
#    plt.title('Variable Importance')
#    plt.show()
    
