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
from sklearn.svm import SVR

# Categorical variables for encoding
categorical = ['category', 'color', 'size', 'channel']
numerical = ['catalog_price', 'original_price', 'unit_price', 'cost_price']

def initialization():
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

    return X, y

def split(X, y, num):
    # new ['category', 'color', 'size', 'channel', 'catalog_price', 'original_price', 'unit_price', 'cost_price']

    # Predicting profit margin for the first test
    # Predict how much quantity bought will change if unit_price changes
    # Predict item total based on original price and channel
    # Need to handle correlation
    feature_indices = {
        0: [0, 1, 2, 3, 4, 7],  # Features for profit margin    
        1: [0, 1, 2, 3, 4, 5],   # Features for quantity     
        2: [0, 1, 2, 3, 4, 6, 7]    # Features for profit
    }
    features = X.iloc[:, feature_indices[num]]
    predicts = y.iloc[:, num]

    return features, predicts


def train_model(X, y):
    np.random.seed(42)
    cv = LeaveOneOut()

    c = [col for col in categorical if col in X.columns]
    n = [col for col in numerical if col in X.columns]

    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # These indices skew the data negatively
        mask = np.ones(len(X_train), dtype=bool)
        mask[[291, 263]] = False
        y_train = y_train[mask]
        X_train = X_train[mask]

        one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        categorical_X_train = one_hot.fit_transform(X_train[c])
        categorical_X_test = one_hot.transform(X_test[c])

        # Apply transformations to skewed data
        yeo_johnson = PowerTransformer(method='yeo-johnson', standardize=True)
        y_train = yeo_johnson.fit_transform(y_train.to_frame())
        y_test = yeo_johnson.transform(y_test.to_frame())

        # Profit margin, quantity, and item total are on different scales
        X_scaler = StandardScaler()

        num_X_train = X_scaler.fit_transform(X_train[n])
        num_X_test = X_scaler.transform(X_test[n])

        X_train = np.hstack([categorical_X_train, num_X_train])
        X_test = np.hstack([categorical_X_test, num_X_test])

        with mlflow.start_run():
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
            model.fit(X_train, y_train)
            mlflow.log_params(model.get_params())
            #DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            #path = os.path.join(DIR, "business_logic", "stats_model.joblib")
            #dump(model, path)

            model = SVR(kernel='rbf', C=1.0, gamma='scale')

            model.fit(X_train, y_train.ravel())

            y_pred = model.predict(X_test)

            # Bring it back to the same dimension
            real_pred = yeo_johnson.inverse_transform(y_pred.reshape(-1, 1))
            real_y_test = yeo_johnson.inverse_transform(y_test)

            mse = mean_squared_error(real_y_test, real_pred)
            #mse = mean_squared_error(y_test, y_pred)
            print(mse)
            print(f'pred: {y_pred}')
            
            #print(f"TRAIN set size: {len(train_index)}")
            #print(f"TEST set size: {len(test_index)}")
            #print(f"mse: {mean_squared_error(y_test, real_pred)}\n")

            '''
            # TO DO: AFTER GETTING QUANTITY PREDICT WITH THE CATALOG PRICE GIVEN IN CUSTOMER_PREDICT
            quantites = real_pred[:, 1]

            for i, j in zip(quantites, X_test[:, 3]):
                print(i * j)

            
            for i, name in enumerate(['profit_margin', 'quantity', 'profit']):
                if name == 'profit_margin':
                    print(f"{name}: {mean_squared_error(real_y_test[:, i], real_pred[:, i])}")
                else:
                    print(f"{name}: {mean_absolute_error(real_y_test[:, i], real_pred[:, i])}")
            #print(mse)

            '''
            # Log metrics and model
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "sales_predict_model")
    #return model

X, y = initialization()

for num in range(3):
    features, predicts = split(X, y, num)
    train_model(features, predicts)



