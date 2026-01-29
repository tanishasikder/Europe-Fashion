import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import uniform, randint
from scipy import stats


'''
product level predictions
Product Success: Predict whether a given product (based on color, size, category, price) 
will sell well in the next period.
Price Sensitivity: Predict whether lowering or raising the price will affect sales volume.
predict the profit based on the product


handle non constant variance find it in any form of data you can
research other stats stuff thats bad in datasets like non constant variance fan shape thing
perhaps do binning for some balance cuz things are approximations

from the data you need to determine which products are
successful, okay, and bad. label stuff like 'successful', 
'fail', etc. have a threshold of sales and profit.

extract features like purchase frequency per product, what
percentage of customers bought a product, do customers 
repeat this purchase, do people who buy product A also buy
product B, does season or time affect the purchases?

heuristic that gives a score to each kind of clothing
1. frequency of purchase
2. money spent
3. original price vs unit price (discount could increase likelihood)
4. if two similar products are different in how theyre bought
5. where the product is being sold
6. production of clothing (maybe more clothes of a certain
category is making it be purchased more or less?)


Price Sensitivity Trends
Question: Does discounting (difference between original_price and unit_price) affect 
purchases in different bins?

Method: Filter products where price was discounted, compare sales in discounted vs 
full-price items for each bin.

Insight Example:

“Medium-priced items saw a 25% boost in sales when discounted, while large-priced 
items showed almost no increase.”
'''


'''
Create a class for MTGBM that solves multi-task learning. Decision trees (LGBM)
are shared then adjusted depending on the task. Parameters are defined in the 
MTGBM class. There are methods to get things within the MTGBM class using self
such as weights, feature_indices, models, etc. 
The clothing_predict method loads in data, initializes dataframes, preprocesses, etc.
Depending on the specific task, different columns will be used to prevent data leakage.
Therefore, feature_indices are used to keep track of these columns, and pass the appropriate
ones to the model. The MTGBM class uses these indices. 
Augmented features uses these indices and keeps correlations that meet a specific
threshold. 
The fit method gets the weights and correlations for the specific task. It loops through
every task and gets the feature_indices and augmentations. With the parameters
that were initialized, it trains a model with lgb and stores the current model, predicts, 
and gets the feature_importance.
'''

# Creating a class for MTGBM with parameters to inherit.
# BaseEstimator and RegressorMixin gives methods
class MTGBM(BaseEstimator, RegressorMixin):
    # Initializing parameters for the later models to use
    def __init__(
        self,
        n_tasks : int = 3,
        n_estimators=50,      
        learning_rate=0.05,    
        max_depth=3,           
        #num_leaves=7,         
        min_child_samples=30,   
        #subsample=0.8,
        #colsample_bytree=0.8,
        random_state=42,
        reg_alpha = 0.5,
        reg_lambda = 3.0,
        share_embeddings: bool = True,  
        verbose: int = -1
        #feature_indices: Dict[int, List[int]] = None 
    ):
        self.n_tasks = n_tasks
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        #self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        #self.subsample = subsample
        #self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.share_embeddings = share_embeddings
        self.verbose = verbose
        #self.feature_indices = feature_indices

        # Initializing list to have the base models
        self.models_: List[lgb.Booster] = []
        # Initializing future weight
        self.task_weights_: np.ndarray = None
        # Initializing any correlations to help the model
        self.task_correlations_: np.ndarray = None
        # Initializing this to see how important features are
        self.feature_importances_: Dict[int, np.ndarray] = {}
        # Initializing storage for the indices used for each feature
        self.feature_indices_: Dict[int, np.ndarray] = None
        # Track how many augmented features there are
        self.n_augmented_features_: Dict[int, int] = {}
    
    # y was (n_samples, n_tasks) but .T flips it so it can
    # be in a correlation matrix
    def _get_correlations(self, y: np.ndarray):
        return np.corrcoef(y.T)
    
    '''
    selected_task = X[:, self.feature_indices_[tasks]]
    augmented = self._augmented_features(selected_task, tasks, task_predictions)

    expected_num = self.n_augmented_features_[tasks]
    actual_num = augmented.shape[1]
    '''
    def _augmented_features(self, X: np.ndarray, task_id: int,
                            other_predictions: Dict[int, np.ndarray] = None):
        # Add predictions from other tasks as features
        if not self.share_embeddings or other_predictions is None:
            return X
        # First initialize the list with the parameter array
        augmented = [X]

        # Loop as long as there are other predictors
        for task_ids, predictors in other_predictions.items():
            if task_ids != task_id:
                # Check the correlation between two different tasks
                correlation = self.task_correlations_[task_id, task_ids]
                # Keep the predictor if it has a specific correlation
                if abs(correlation) > 0.3:
                    kept_pred = predictors.reshape(-1, 1) * abs(correlation)
                    augmented.append(kept_pred)
         
        return np.hstack(augmented)
    
    # y is an array for (n_samples, n_tasks)
    def _get_task_weights(self, y: np.ndarray):
        variances = np.var(y, axis=0)
        weights = 1.0 / (variances + 1e-8)
        return weights / weights.sum()
    
    def fit(self, X : np.ndarray, y: np.ndarray, feature_indices):
        # X is a list of the features per model. Within each list
        # (n_samples, n_features) are the kinds of values
        # y is (n_samples, n_tasks) which are the targets per task

        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        if y.shape[1] != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} tasks, got {y.shape[1]}")

        # Store feature indices
        if self.feature_indices_ is None:
            # Use all features for all tasks
            self.feature_indices_ = feature_indices

        self.task_weights_ = self._get_task_weights(y)
        self.task_correlations_ = self._get_correlations(y)

        self.models_ = []
        task_predictions = {}

        self.n_augmented_features_ = {}

        for task in range(self.n_tasks):
            # Depending on the task, different features will be used
            specific_task = X[:, self.feature_indices_[task]]
            # Gets augmented features based on the current task, feature_indices, and predictions
            augmented = self._augmented_features(specific_task, task, task_predictions)
            # Training dataset based on augmented features and labels from the targets
            train = lgb.Dataset(augmented, label=y[:, task])

            parameters = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate' : self.learning_rate,
                'max_depth' : self.max_depth,
                #'num_leaves' : self.num_leaves,
                'min_child_samples' : self.min_child_samples,
                #'subsample' : self.subsample,
                #'colsample_bytree' : self.colsample_bytree,
                'random_state' : self.random_state,
                'reg_alpha' : self.reg_alpha,
                'reg_lambda' : self.reg_lambda,   
            }

            model = lgb.train(
                parameters,
                train,
                num_boost_round=self.n_estimators
            )

            # Stores the model, predicts with the augmented features,
            # and stores the model's feature importance
            self.models_.append(model)
            task_predictions[task] = model.predict(augmented)
            self.feature_importances_[task] = model.feature_importance()
            self.n_augmented_features_[task] = augmented.shape[1]

        return self

    def predict(self, X):
        # Predicting all tasks at once

        # Initially create empty array/dictionary
        predictions = np.zeros((X.shape[0], self.n_tasks))
        task_predictions = {}

        # Loop through every task and pass it through previous function
        for tasks in range(self.n_tasks):
            selected_task = X[:, self.feature_indices_[tasks]]
            augmented = self._augmented_features(selected_task, tasks, task_predictions)

            expected_num = self.n_augmented_features_[tasks]
            actual_num = augmented.shape[1]
            # If expected count and actual count don't match then pad with zeros
            # Sometimes only keeping values with a specific correlation can make the
            # n_features not be the same as the original n_features
            if actual_num < expected_num:
                # Creates extra columns of zeros so errors don't happen
                padding = np.zeros((selected_task.shape[0], expected_num - actual_num))
                augmented = np.hstack([augmented, padding])

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

        # Save product_compare as a CSV file to generate synthetic data
        #product_compare.to_csv('C:/Users/Tanis/Downloads/Europe-Fashion/Fashion Data/product_comparing.csv', index=False)

        # After making the CSV file, generate synthetic data, loading here.
        synthetic = pd.read_csv('Fashion Data/synthetic_data.csv')

        # Combine synthetic data with original data
        product_compare = pd.concat([product_compare, synthetic], ignore_index=True)

        # Adds a column to show the product's cost to make
        product_compare['cost_to_make'] = product_compare['cost_price'] * product_compare['quantity']

        # Adds a column to show profit
        product_compare['profit'] = product_compare['item_total'] - product_compare['cost_to_make']

        #Adds a column to show profit margin
        product_compare['profit_margin'] = (product_compare['unit_price'] - product_compare['cost_price']) / product_compare['unit_price']

        # Categorical variables for encoding
        categorical = ['category', 'color', 'size', 'channel']

        categorical_encoding = {}

        for col in categorical:
            label = LabelEncoder()
            # In the initial dataframe, replace every categorical value with the encoded one
            product_compare[col] = label.fit_transform(product_compare[col])
            # Store the encoders used for every categorical variable
            categorical_encoding[col] = label

        X = product_compare[['category', 'color', 'size', 'catalog_price', 'channel', 'original_price', 'unit_price']].values
        y = product_compare[['profit_margin', 'quantity', 'item_total']].values
        
        #X[:, numeric_cols] = scaler.fit_transform(X[:, numeric_cols])

        '''
        scaled_columns = [0, 1, 2, 3, 4, 5, 6]

        scaler = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), scaled_columns)
            ],
            remainder='passthrough'
        )
        '''

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        '''
        gb = GradientBoostingRegressor(random_state=42)
        multi = MultiOutputRegressor(gb)
        grid = GridSearchCV(multi, params, cv=5, scoring='neg_mean_absolute_error')
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        print(f"Best score: {-grid.best_score_}")
        '''
        # These indices skew the data negatively
        mask = np.ones(len(X_train), dtype=bool)
        mask[[291, 263]] = False
        y_train = y_train[mask]
        X_train = X_train[mask]

       # X_train = scaler.fit_transform(X_train)
        #X_test = scaler.transform(X_test)

        #['category', 'color', 'size', 'catalog_price', 'channel', 'original_price', 'unit_price']
        # Predicting profit margin for the first test
        # Predict how much quantity bought will change if unit_price changes
        # Predict item total based on original price and channel
        # Need to handle correlation
        feature_indices = {
            0: [0, 1, 2, 3],  # Features for profit margin    
            1: [0, 1, 2, 3, 5],   # Features for quantity     
            2: [0, 1, 2, 4, 5]    # Features for item_total
        }

        model = MTGBM(
            n_tasks=3,
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            #share_embeddings=True,
            verbose=-1,
            random_state=42,
            #objective='huber'
        )

        model.fit(X_train, y_train, feature_indices)

        pred = model.predict(X_test)

        # The things that will be predicted
        task_names = ['profit_margin', 'quantity', 'item_total']

        for task_id in range(3):
            print(f"TASK {task_id}: {task_names[task_id]}")
            y_true = y_test[:, task_id]
            y_task_pred = pred[:, task_id]

            mse = mean_absolute_error(y_true, y_task_pred)
            print(f"Mean Squared Error: {mse}")
            
        
        # Predict if the product will sell at full price (classification)
        product_compare['sold_full_price'] = (
            product_compare['unit_price'] >= product_compare['catalog_price'] * 0.95
        )

        

clothing_predict()