### Thomas Hepner
### 6/18/2016
### Facebook: Identify the correct place for check-ins

# Import libraries
import os
import numpy as np
import pandas as pd

def load_data(): 
    """ Load competition data. 
    """
    ### Set work directory
    os.chdir('C://Users//thep3//OneDrive//Documents//Kaggle//Facebook V - Predicting Check Ins//data//')
    
    ### Load data
    train = pd.read_csv("train.csv", sep = ",") 
    test = pd.read_csv("test.csv", sep = ",")
    
    # Return data
    return train, test

def transform_data(train, test):
    """ Transform train and test data to include new variables.
    """
    
    # Time
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')  # Arbitrary date chosen
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in train['time'].values)    
    train['hour'] = d_times.hour
    train['weekday'] = d_times.weekday
    train['day_of_month'] = d_times.day
    train['month'] = d_times.month
    train['year'] = d_times.year

    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') for mn in test['time'].values)    
    test['hour'] = d_times.hour
    test['weekday'] = d_times.weekday
    test['day_of_month'] = d_times.day
    test['month'] = d_times.month
    test['year'] = d_times.year  
            
    # Accuracy 
    train['accuracy'] = np.log10(train['accuracy']) * 10.0
    test['accuracy'] = np.log10(test['accuracy']) * 10.0
      
    # Combine x and y attributes
    eps = 0.00001  
    train['x_d_y'] = train.x.values / (train.y.values + eps) 
    test['x_d_y'] = test.x.values / (test.y.values + eps)    
    
    train['x_t_y'] = train.x.values * train.y.values 
    test['x_t_y'] = test.x.values * test.y.values 
            
    # Return data
    return train, test  

def add_grid_cell_variables(train, test, grid_sizes):
    """ Adds new grid cell variables for each grid size to train and test data.
    """
    # Create new variables according to size of grids
    for grid in grid_sizes:
        # Define grid boundaries
        n_cell_x = grid[0]
        n_cell_y = grid[1]
        size_x = 10.0 / n_cell_x
        size_y = 10.0 / n_cell_y
        eps = 0.00001  
        
        # Add grid variables to train data
        xs = np.where(train.x.values < eps, 0, train.x.values - eps)
        ys = np.where(train.y.values < eps, 0, train.y.values - eps)
        pos_x = (xs / size_x).astype(np.int)
        pos_y = (ys / size_y).astype(np.int)
        variable_name = 'grid_cell_' + str(n_cell_x) + 'x' + str(n_cell_y)
        train[variable_name] = pos_y * n_cell_x + pos_x
                        
        # Add grid variables to test data
        xs = np.where(test.x.values < eps, 0, test.x.values - eps)
        ys = np.where(test.y.values < eps, 0, test.y.values - eps)
        pos_x = (xs / size_x).astype(np.int)
        pos_y = (ys / size_y).astype(np.int)
        test[variable_name] = pos_y * n_cell_x + pos_x
        
    # Return data
    return train, test
    
def normalize_data(train, test):
    """ Scales and centers data of numerical variables.
    """
    
    # Normalize data of categorical variables
    columns = ['x', 'y', 'accuracy', 'x_d_y', 'x_t_y', 'hour', 'weekday', 'day_of_month', 'month', 'year']
    for cl in columns:
        ave = train[cl].mean()
        std = train[cl].std()
        train[cl] = (train[cl].values - ave ) / std
        test[cl] = (test[cl].values - ave ) / std
     
    # Drop unnecessary variables
    train = train.drop(['row_id', 'time'], axis = 1)
    test = test.drop(['row_id', 'time'], axis = 1) 

    # Return data
    return train, test    
    
#### Execute Code #### 
if __name__ == '__main__':
    """ Builds data files needed to create models and generate predictions.
    """
    
    print "1. Loading data..."
    train, test = load_data()
    
    print "2. Transforming data..."
    train, test = transform_data(train, test)
    
    print "3. Adding grid variables to data..."
    grid_sizes = [(20, 40), (50, 50), (100, 100), (50, 100), (75, 150), (100, 200)]
    train, test = add_grid_cell_variables(train, test, grid_sizes)
    
    print "4. Normalizing data..."
    train, test = normalize_data(train, test)
    
    print "4. Writing to csv..."
    train.to_csv("train_modified.csv", index = False, header = True)
    test.to_csv("test_modified.csv", index = False, header = True)
    
    print "5. Complete!"

    
    