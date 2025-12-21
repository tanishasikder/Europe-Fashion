import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Creating a class for MTGBM with parameters to inherit.
# BaseEstimator and RegressorMixin gives methods
class MTGBM(BaseEstimator, RegressorMixin):
    # Initializing parameters for the later models to use
    def __init__(
        self,
        n_tasks = int,
        n_estimators=100,      
        learning_rate=0.05,    
        max_depth=3,           
        num_leaves=15,         
        min_child_samples=5,   
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha = 0.1,
        reg_lambda = 0.1
    ):
        self.n_tasks = n_tasks
        self.n_estimators = n_estimators
        self.n_learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        # Initializing list to have the base models
        self.models_: List[lgb.Booster] = []
        # Initializing future weight
        self.task_weights_: np.ndarray = None
        # Initializing any correlations to help the model
        self.task_correlations_: np.ndarray = None
        # Initializing this to see how important features are
        self.feature_importances_: Dict[int, np.ndarray] = {}

    # y is an array for (n_samples, n_tasks)
    def _get_task_weights(self, y: np.ndarray):
        variances = np.var(y, axis=0)
        weights = 1.0 / (variances + 1e-8)
        return weights / weights.sum()
    
    # y was (n_samples, n_tasks) but .T flips it so it can
    # be in a correlation matrix
    def _get_correlations(self, y: np.ndarray):
        return np.corrcoef(y.T)

    def _augmented_features(self, X: np.ndarray, task_id: int,
                            other_predictions: Dict[int, np.cdarray] = None):
        # Add predictions from other tasks as features

        # First initialize the list with the parameter array
        augmented = [X]

        # Loop as long as there are other predictors
        if len(other_predictions) > 0:
            for task_ids, predictors in other_predictions.items():
                if task_ids != task_id:
                    correlation = self.task_correlations_[task_id, task_ids]
                    if abs(correlation) > 0.4:
                        kept_pred = predictors.reshape(-1, 1) * abs(correlation)
                        augmented.append(kept_pred)
        
        return np.hstack(augmented)
    
    def fit_model(self, X : List[np.ndarray], y: np.ndarray, categorical):
        # X is a list of the features per model. Within each list
        # (n_samples, n_features) are the kinds of values
        # y is (n_samples, n_tasks) which are the targets per task

        self.task_weights_ = self._get_task_weights(y)
        self.task_correlations_ = self._get_correlations(y)

        self.models_ = []
        task_predictions = {}

        encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

        # Making a pipeline
        preprocess = ColumnTransformer(transformers=[
            ('onehot', encoder, categorical),
            ('numerical', Pipeline([
                ('power', PowerTransformer(method='yeo-johnson'),
                 ('scaler', StandardScaler()))
            ]))
        ])

        for feature_list in X:
            current_models = []
            current_predictions = {}

            for task in range(self.n_tasks):
                augmented = self._augmented_features(feature_list, task, task_predictions)

                parameters = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'n_tasks' : self.n_tasks,
                    'n_estimators' : self.n_estimators,
                    'learning_rate' : self.n_learning_rate,
                    'max_depth' : self.max_depth,
                    'num_leaves' : self.num_leaves,
                    'min_child_samples' : self.min_child_samples,
                    'subsample' : self.subsample,
                    'colsample_bytree' : self.colsample_bytree,
                    'random_state' : self.random_state,
                    'reg_alpha' : self.reg_alpha,
                    'reg_alpha' : self.reg_lambda,   
                }

                model = Pipeline(steps=[
                    ('preprocessing', preprocess),
                    ('lgbm', lgb.LGBMRegressor(parameters))
                ])

                model.fit(augmented, y[:, task])
                current_models.append(model)
                current_predictions[task] = model.predict(augmented)

            self.models_ = current_models
            task_predictions = current_predictions
        
        return self

    def predict(self, X: np.ndarray):
        # Predicting all tasks at once

        # Initially create empty array/dictionary
        predictions = np.zeros((X.shape[0], self.n_tasks))
        task_predictions = {}

        # Loop through every task and pass it through previous function
        for tasks in range(self.n_tasks):
            augmented = self._augmented_features(X, tasks, task_predictions)
            predict = self.models_[tasks].predict(augmented)
            predictions[:, tasks] = predict
            task_predictions[tasks] = predict
        
        return predictions
    
    def clothing_predict():
        np.random.seed(42)
        data = pd.read_excel('Fashion Data/DataPenjualanFashion.xlsx', sheet_name=None)

        # Separating the data into dataframes
        products = data['ProductItems']
        sales_items = data['SalesItems']
        pivot = data['Pivot Table']

        # Cleaning the data
        # Reorganizing pivot
        channel_data = {'Channels' : ['App Mobile', 'E-commerce'],
                        'Totals Of Original Price' : [53952.79, 57167.84]}

        types_data = {'Type Product' : ['Dresses', 'Pants', 'Shoes', 'Sleepwear', 'T-Shirt', 'Grand Total'],
                'Total Catalog Price' : [5298.44, 3959.36, 5236.03, 4933.21, 5511.98, 24939.02], 
                'Total Cost Price' : [2917.77, 2149.76, 2908.44, 2785.75, 2962.69, 13724.41]}

        # Shows the money by the channels
        channel = pd.DataFrame(channel_data)

        # Shows each category of clothing and their listed price and cost to make
        types = pd.DataFrame(types_data)

        pivot = pivot.dropna()
        pivot = pivot.drop(19)
        # Pivot now just shows the categories and color of the clothing options
        pivot = pivot.rename(columns={'Unnamed: 0' : 'Row Labels', 'Unnamed: 1' : 'Dresses', 'Unnamed: 2' : 'Pants', 
                            'Unnamed: 3' : 'Shoes','Unnamed: 4' : 'Sleepwear', 'Unnamed: 5' : 'T-Shirts', 
                            'Unnamed: 6' : 'Grand Total'})

        # Dataframe for the total cost to make the items and how much customers spent for the items
        product_compare = sales_items.merge(
            products,
            left_index=True,
            right_index=True
        )

        # Adds a column to show the product's cost to make
        product_compare['cost_to_make'] = product_compare['cost_price'] * product_compare['quantity']

        # Adds a column to show profit
        product_compare['profit'] = product_compare['item_total'] - product_compare['cost_to_make']

        #Adds a column to show profit margin
        product_compare['profit_margin'] = (product_compare['unit_price'] - product_compare['cost_price']) / product_compare['unit_price']

        # Categorical variables for encoding
        categorical = ['category', 'color', 'size', 'channel']

        # Predicting profit margin for the first test
        first_test = product_compare['category', 'color', 'size', 'catalog_price', 'channel']
        
        # Predict how much quantity bought will change if unit_price changes
        second_test = sales_items['original_price', 'unit_price']
        
        # Predict how much of a discount will increase/decrease profit margins
        
        # Predict item total based on original price and channel
        fourth_test = sales_items['original_price', 'channel']

        first_data = product_compare['profit_margin']
        more_data = sales_items['quantity', 'item_total']

        X = np.column_stack([first_test, second_test, fourth_test])
        y = np.column_stack([first_data, more_data])


        # Predict if the product will sell at full price (classification)
        product_compare['sold_full_price'] = (
            product_compare['unit_price'] >= product_compare['catalog_price'] * 0.95
        )
