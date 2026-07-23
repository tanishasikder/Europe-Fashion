import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from pathlib import Path
import yaml
from dotenv import load_dotenv
import dagshub
import os

# Best Practice for locating parent directories
BASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR1 = Path(__file__).resolve().parents[2]
# Safely join directories/files using the division operator
CONFIG_PATH = BASE_DIR / 'config'/'training_config.yaml'

# Categorical variables for encoding
categorical = ['category', 'color', 'size', 'channel']
numerical = ['catalog_price', 'original_price', 'unit_price', 'cost_price']

def initialization():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = BASE_DIR1 / config['dataset']['raw_path']
    #print(f'data path is this: {data_path}')
    data = pd.read_excel(data_path, sheet_name=None)

    # Separating the data into dataframes
    products = data['ProductItems']
    sales_items = data['SalesItems']

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
        0: [0, 1, 2, 4, 7],  # Features for profit margin    
        1: [0, 1, 2, 4],   # Features for quantity     
        2: [0, 1, 2, 4, 7]    # Features for profit
    }
    features = X.iloc[:, feature_indices[num]]
    predicts = y.iloc[:, num]

    return features, predicts


def train_model(X, y):
    np.random.seed(42)

    c = [col for col in categorical if col in X.columns]
    n = [col for col in numerical if col in X.columns]

    greatest_mse = float('inf')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

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

        model.fit(X_train, y_train.ravel())
        mlflow.log_params(model.get_params())
        y_pred = model.predict(X_test)

        # Bring it back to the same dimension
        real_pred = yeo_johnson.inverse_transform(y_pred.reshape(-1, 1))
        real_y_test = yeo_johnson.inverse_transform(y_test)

        mse = mean_squared_error(real_y_test, real_pred)

        if mse < greatest_mse:
            # Log metrics and model
            greatest_mse = mse
            mlflow.log_metric("mse", mse)
            model_info = mlflow.sklearn.log_model(model, name="europe-fashion-sales-model")
            mlflow.register_model(model_uri=f"models:/{model_info.model_id}", name="europe-fashion-sales-model")
        mlflow.end_run()

if __name__ == "__main__":
    OWNER = os.getenv('DAGSHUB_OWNER')
    REPO = os.getenv('DAGSHUB_REPO')

    dagshub.init(repo_owner=OWNER, repo_name=REPO, mlflow=True)

    X, y = initialization()

    for num in range(3):
        features, predicts = split(X, y, num)
        results = train_model(features, predicts)




