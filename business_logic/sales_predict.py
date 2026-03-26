import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split, cross_val_score
from typing import List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# Creating a class for MTGBM with parameters to inherit.
# BaseEstimator and RegressorMixin gives methods
class MTGBM(BaseEstimator, RegressorMixin):
    # Initializing parameters for the later models to use
    def __init__(
        self,
        n_tasks : int = 3,
        n_estimators=50,      
        learning_rate=0.05,    
        max_depth=4,                   
        min_child_samples=32,   
        random_state=41,
        reg_lambda = 5.0,
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
    
    # Parameters are feature indices, specific task, and task predictions
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

    def predict_values(self, user_params):
        # Predicting all tasks at once
        # Initially create empty array/dictionary
        predictions = np.zeros((len(user_params), self.n_tasks))
        task_predictions = {}

        for task in range(self.n_tasks):
            # User selects a specific task and gives formatted features
            specific_task = user_params[:, self.feature_indices_[task]]
            # Augmented is made based on these
            augmented = self._augmented_features(specific_task, task, task_predictions)
 
            expected_num = self.n_augmented_features_[task]
            actual_num = augmented.shape[1]
            # If expected count and actual count don't match then pad with zeros
            # Sometimes only keeping values with a specific correlation can make the
            # n_features not be the same as the original n_features
            if actual_num < expected_num:
                # Creates extra columns of zeros so errors don't happen
                padding = np.zeros((user_params[0], expected_num - actual_num))
                augmented = np.hstack([augmented, padding])

            task_predictions[task] = self.models_[task].predict(augmented)
            predictions[:, task] = task_predictions[task]
    
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
        #synthetic = pd.read_csv('Fashion Data/synthetic_data.csv')
        
        #synthetic_sample = synthetic.sample(n=100, random_state=42)
        #Combine synthetic data with original data
        #product_compare = pd.concat([product_compare, synthetic_sample], ignore_index=True)

        # Adds a column to show the product's cost to make
        product_compare['cost_to_make'] = product_compare['cost_price'] * product_compare['quantity']

        # Adds a column to show profit
        product_compare['profit'] = product_compare['item_total'] - product_compare['cost_to_make']

        #Adds a column to show profit margin
        product_compare['profit_margin'] = (product_compare['unit_price'] - product_compare['cost_price']) / product_compare['unit_price']
        #product_compare = product_compare[product_compare['profit_margin'] >= 0]
        
        X = product_compare[['category', 'color', 'size', 'channel', 'catalog_price', 'original_price', 'unit_price', 'cost_price']]
        y = product_compare[['profit_margin', 'quantity', 'profit']]

        cv = LeaveOneOut()

        # Categorical variables for encoding
        categorical = ['category', 'color', 'size', 'channel']
        numerical = ['catalog_price', 'original_price', 'unit_price', 'cost_price']

        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # These indices skew the data negatively
            mask = np.ones(len(X_train), dtype=bool)
            mask[[291, 263]] = False
            y_train = y_train[mask]
            X_train = X_train[mask]

            one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            
            categorical_X_train = one_hot.fit_transform(X_train[categorical])
            categorical_X_test = one_hot.transform(X_test[categorical])

            # Profit margin, quantity, and item total are on different scales
            X_scaler = StandardScaler()

            num_X_train = X_scaler.fit_transform(X_train[numerical])
            num_X_test = X_scaler.transform(X_test[numerical])

            X_train = np.hstack([categorical_X_train, num_X_train])
            X_test = np.hstack([categorical_X_test, num_X_test])

            # Apply transformations to skewed data
            yeo_johnson = PowerTransformer(method='yeo-johnson', standardize=True)

            y_train = yeo_johnson.fit_transform(y_train)
            y_test = yeo_johnson.transform(y_test)


            # new ['category', 'color', 'size', 'channel', 'catalog_price', 'original_price', 'unit_price', 'cost_price']
            # old ['category', 'color', 'size', 'catalog_price', 'channel', 'original_price', 'unit_price', 'cost_price']

            # Predicting profit margin for the first test
            # Predict how much quantity bought will change if unit_price changes
            # Predict item total based on original price and channel
            # Need to handle correlation
            feature_indices = {
                0: [0, 1, 2, 3, 4, 7],  # Features for profit margin    
                1: [0, 1, 2, 3, 4, 5],   # Features for quantity     
                2: [0, 1, 2, 3, 4, 6, 7]    # Features for profit
            }

            with mlflow.start_run():
                model = MTGBM(
                    n_tasks=3,
                    n_estimators=50,
                    learning_rate=0.05,
                    max_depth=4,
                    verbose=-1,
                    random_state=42
                )
                
                model.fit(X_train, y_train, feature_indices)
                mlflow.log_params(model.get_params())
                #DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                #path = os.path.join(DIR, "business_logic", "stats_model.joblib")
                #dump(model, path)

                # Make the model predict then test the metrics
                pred = model.predict_values(X_test)
                # Bring it back to the same dimension
                real_pred = yeo_johnson.inverse_transform(pred)
                real_y_test = yeo_johnson.inverse_transform(y_test)
                #print(f"TRAIN set size: {len(train_index)}")
                #print(f"TEST set size: {len(test_index)}")
                #print(f"mse: {mean_squared_error(y_test, real_pred)}\n")

                '''
                # TO DO: AFTER GETTING QUANTITY PREDICT WITH THE CATALOG PRICE GIVEN IN CUSTOMER_PREDICT
                quantites = real_pred[:, 1]

                for i, j in zip(quantites, X_test[:, 3]):
                    print(i * j)

                '''
                for i, name in enumerate(['profit_margin', 'quantity', 'profit']):
                    if name == 'profit_margin':
                        print(f"{name}: {mean_squared_error(real_y_test[:, i], real_pred[:, i])}")
                    else:
                        print(f"{name}: {mean_absolute_error(real_y_test[:, i], real_pred[:, i])}")
                #print(mse)

                
                # Log metrics and model
                #mlflow.log_metric("mse", mse)
                #mlflow.sklearn.log_model(model, "sales_predict_model")
        #return model

mtgbm = MTGBM()
mtgbm.train_model()




