Thomas Hepner, July 2016: 

Project code for the Facebook V - Predicting Check Ins
Kaggle competition that I used to place 87th out of 1212
competitors. 

The code is broken up into the following files/functions: 

1. build: Loads, cleans, and transforms data for model building process. 

2. tune: Sample 0.5% of data and builds predictive models.
	- tune_nn: Tune Nearest Neighbor model.
	- tune_nn_features: Finds optimal feature weights for Nearest Neighbor model.
	- tune_tree models: Tunes XGBOOST and Random Forest models.

3. predict: 
	- predict: Generates test data predictions for tree models.
	- predict_nn: Generate test data predictions for Nearest Neighbor model.

4. combine_predictions:
	- Ensembles test data predictions from multiple models 
	  using Mean-Average Precision (MAP3) weighting for top 3 predictions.
	