### Thomas Hepner
### 6/18/2016
### Facebook: Identify the correct place for check-ins

# Import general libraries
import os
import time
import math

# Import statistical and machine learning libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def MAP3(actuals, predictions):
    """ Mean Average Precision metric used for evaluation.
    """
    n = len(actuals)
    average_precisions = [0] * n
    for i in range(0, n): 
        correct = actuals[i]
        precision_k = 0.0
        u = len(predictions[i])
        for j in range(0, u):                
            if(predictions[i][j] == correct):
                precision_k = 1.0 / (j + 1)
                average_precisions[i] = precision_k
                break
    map3 = np.asarray(average_precisions).mean()
    return map3

def load_data(): 
    """ Load competition data. 
    """
    ### Set work directory
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//data//')
    
    ### Load data
    train = pd.read_csv("train_modified.csv", sep = ",") 
    test = pd.read_csv("test_modified.csv", sep = ",")
    
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
#    train['accuracy'] *= weights[7]
#    train['x_d_y'] *= weights[8]
#    train['x_t_y'] *= weights[9]

    # Transform test data
    test['x'] *= weights[0]    
    test['y'] *= weights[1]       
    test['hour'] *= weights[2] 
    test['weekday'] *= weights[3]
    test['day_of_month'] *= weights[4]
    test['month'] *= weights[5]
    test['year'] *= weights[6]    
#    test['accuracy'] *= weights[7]
#    test['x_d_y'] *= weights[8]
#    test['x_t_y'] *= weights[9]
    
    # Return transformed data
    return train, test
    
def select_grid_variable(train, test, grid_variable):
    """ Removes other grid variables from data. 
    """
    grid_variables = ['grid_cell_20x40', 'grid_cell_50x50', 'grid_cell_100x100', 'grid_cell_50x100', 'grid_cell_75x150', 'grid_cell_100x200']
    grid_variables.remove(grid_variable)    
    
    train = train.drop(grid_variables, 1)    
    test = test.drop(grid_variables, 1)
    
    # Return data
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
    X_test = test_cell.drop([grid_variable], axis = 1).values.astype(int)
    
    # Build training classifier and predict
    model.fit(X_train, y_train)
    X_test = test_cell.drop([grid_variable], axis = 1).values
    y_pred = model.predict_proba(X_test)

    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])   
    end = time.time()
    time_elapsed = (end - start)
    
    # Return data
    return pred_labels, test_ids, time_elapsed

def process_grid(train, test, threshold, model, model_name, grid_variable): 
    """ Iterates through all cells in grid, builds classifiers, and generates predictions. 
        Outputs submission.
    """            
    # Iterates through grid, building a predictive model for each grid cell
    preds = np.zeros((test.shape[0], 3), dtype=int)
    preds = preds.astype(str)
    execution_time = 0
    num_cells = len(train[grid_variable].unique())
    for grid_id in range(num_cells):
        # Applying classifier to one grid cell
        pred_labels, test_ids, time_elapsed = process_grid_cell(train, test, grid_id, threshold, model, grid_variable)

        # Print percentage of process completed
        execution_time = execution_time + time_elapsed
        print "    " + str(round(100.0 * grid_id / num_cells, 2)) + '% complete: ' + str(round(execution_time, 3)) + ' seconds'

        # Updating predictions
        preds[test_ids] = pred_labels.astype(str)

    # Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype = str, columns = ['l1', 'l2', 'l3'])  
    
    # Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep = ' ')
    
    # Write to csv
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//submissions')
    ds_sub.name = 'place_id'
    name = 'submission_' + model_name + '.csv'
    ds_sub.to_csv(name, index = True, header = True, index_label = 'row_id')
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//submissions//')    
                                    
#### Execute Code ####   
if __name__ == '__main__':
    """ Generates place_id label predictions for test data.
    """
    
    print "1. Loading data..."
    train, test = load_data()
    
    print "2. Transforming data..."
    feature_weights = [200, 580, 5, 2, 1, 2, 1, 2, 100, 100]
    train, test = transform_data(train, test, feature_weights)
    
    print "3. Removing select appropriate grid variable..."
    grid_variable = 'grid_cell_50x100'
    train, test = select_grid_variable(train, test, grid_variable = grid_variable)
    
    print "4. Building models and generating predictions..."
    model_nn = KNeighborsClassifier(n_neighbors = 25, n_jobs = -1, weights = 'distance', metric = 'manhattan')
    nn_threshold = 5
    process_grid(train, test, threshold = nn_threshold, model = model_nn, model_name = 'nn_thresh5_neighbors25_50x100_tuned_weights_v2', grid_variable = grid_variable)
    
    print ""
    print "4. Complete!!!"    
    
