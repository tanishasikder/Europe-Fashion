import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
import pickle
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
    

#mtgbm = MTGBM()
#mtgbm.train_model()




