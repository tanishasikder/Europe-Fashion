import os
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
from joblib import dump, load

from sales_predict import MTGBM

def train_model():
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
        DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(DIR, "business_logic", "stats_model.joblib")
        dump(model, path)

        return model

train_model()