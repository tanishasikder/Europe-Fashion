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
import seaborn as sns
import matplotlib.pyplot as plt

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
        min_child_samples=32,   
        random_state=41,
        reg_lambda = 3.0,
        verbose: int = -1
    ):
        self.n_tasks = n_tasks
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.reg_lambda = reg_lambda
        self.verbose = verbose

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
    
    def _augmented_features(self, X: np.ndarray, task_id: int,
                            other_predictions: Dict[int, np.ndarray] = None):
        # Add predictions from other tasks as features
       # if not self.share_embeddings or other_predictions is None:
        #    return X
        if other_predictions is None:
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
                'min_child_samples' : self.min_child_samples,
                'random_state' : self.random_state,
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

    def predict(self, user_params, select):
        # Predicting all tasks at once

        # Initially create empty array/dictionary
        predictions = np.zeros((1, self.n_tasks))
        task_predictions = {}

        # User selects a specific task and gives formatted features
        # Augmented is made based on these
        augmented = self._augmented_features(user_params, select, task_predictions)

        expected_num = self.n_augmented_features_[select]
        actual_num = augmented.shape[1]
        # If expected count and actual count don't match then pad with zeros
        # Sometimes only keeping values with a specific correlation can make the
        # n_features not be the same as the original n_features
        if actual_num < expected_num:
            # Creates extra columns of zeros so errors don't happen
            padding = np.zeros((user_params, expected_num - actual_num))
            augmented = np.hstack([augmented, padding])

        predict = self.models_[select].predict(augmented)
        predictions[:, select] = predict
        task_predictions[select] = predict
        
        return predictions
    
    def train_model(self):
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
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # These indices skew the data negatively
            mask = np.ones(len(X_train), dtype=bool)
            mask[[291, 263]] = False
            y_train = y_train[mask]
            X_train = X_train[mask]

            #['category', 'color', 'size', 'catalog_price', 'channel', 'original_price', 'unit_price']
            # Predicting profit margin for the first test
            # Predict how much quantity bought will change if unit_price changes
            # Predict item total based on original price and channel
            # Need to handle correlation
            feature_indices = {
                0: [0, 1, 2, 3],  # Features for profit margin    
                1: [0, 1, 2, 3, 5],   # Features for quantity     
                2: [0, 1, 2, 3, 5]    # Features for item_total
            }

            model = MTGBM(
                n_tasks=3,
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                verbose=-1,
                random_state=42
            )

            model.fit(X_train, y_train, feature_indices)

            return model
    
    def clothing_predict(self, user_selection):
        # Fix this later, do not train model everytime
        model = self.train_model()

        task_names = ['profit_margin', 'quantity', 'item_total']

        '''
        user_selection is a number. 0, 1, 2 depending on which task

        user_params is the specific parameters for the task the user gave
        it is in a 2d list

        need to encode categorical variables
        '''
        pred = model.predict(user_selection)
        print(pred)






