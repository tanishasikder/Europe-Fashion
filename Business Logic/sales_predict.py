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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Loading in the data
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
sales_sum = sales_items.groupby('original_price').agg(
    quantity = ('quantity', 'sum'),
    # Was customer_spent
    item_total =('item_total', 'sum')
).reset_index()

cost = products['cost_price']

# Combines the two dataframes
product_compare = pd.merge(
    sales_sum,
    cost,
    left_index=True,
    right_index=True
)


#product_compare['customer_spent'] = product_compare['item_total']
# Adds a column to show the product's cost to make
product_compare['cost_to_make'] = product_compare['cost_price'] * product_compare['quantity']

# Adds a column to show profit
product_compare['profit'] = product_compare['item_total'] - product_compare['cost_to_make']

print(product_compare)


# Dataframe for the cost to make items by category (color and item type) and how much customers spent
total = sales_items.merge(
    products,
    left_index=True,
    right_index=True
)

total['cost_to_make'] = total['cost_price'] * total['quantity']

# GENERATE SIGNIFICANT DATA IF NEEDED
# FIX THIS ALL LATER THERES DATA LEAKAGE CUZ YOURE GROUPING WITH CATEGORY AND COLOUR
# this dataframe is for the categories to be as one. the last one has individual points
# this one has everything aggregated to see how the overall category does
total_compare = total.groupby(['category', 'color']).agg(
    sold_quantity=('quantity', 'sum'),
    item_total=('item_total', 'sum'),
    cost_to_make=('cost_to_make', 'sum')
).reset_index()

total_compare['profit'] = total_compare['item_total'] - total_compare['cost_to_make']
total_compare.sort_values(by=['profit'], ascending=False)

print("_________________________________________________________________")
print(products)
print(product_compare)

# Need to transform this data
skewed_data = []   #ORIGINAL PRICE ISNT IN ANY OF THE DATAFRAMES WHY ARE YOU TRANSFORMING THIS
#skewed_data = ['original_price']

# Need to handle multicolinearity (from one hot encoding)
unnatural_correlation = ['channel', 'channel_campaigns']

# Need to handle multicolinearity (from natural occurences)
natural_correlation = ['original_price', 'unit_price']

# Categorical variables for encoding
categorical = ['category', 'color', 'size']

# Making a pipeline
power_transformer = Pipeline(steps=[
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])


encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
pca = PCA(n_components=2)

# Preprocessing that handles skewed data, encodes, and does dimensionality reduction
preprocess = ColumnTransformer(
    transformers=[
        ('skew_fix', power_transformer, skewed_data),
        ('onehot', encoder, categorical)
        #,('pca', pca, natural_correlation)
    ],
    remainder='passthrough' # Keeps any other columns unchanged
)

# We are using the catalog price now to help predict along with size
# Do train test split here then create an ML pipeline
X = products.drop(['cost_price', 'gender', 'product_id', 'brand', 'product_name'], axis=1)
y = product_compare['profit']

# Drop rows in X to make X and y have the same number of samples
X = X.reindex(y.index)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

negative_indices = y_train[y_train < 0].index.tolist()

# These indices have negative values
y_train = y_train.drop([291, 263], axis=0)
X_train = X_train.drop([291, 263], axis=0)

# Reset indices for alignment
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Log transformation (check to see if it works)
#y_train = np.log(y_train)

linear = LinearRegression()
lgbm = lgb.LGBMRegressor()

# Pipeline that combines preprocessing steps and the linear model
linear_pipeline = Pipeline(steps=[
    ('preprocessing', preprocess),
    ('linear_model', linear)
])

# Test if there are 'continuous' errors

print("y_test dtype:", getattr(y_train, "dtype", type(y_train)))
print("y_test sample:", np.array(y_test[:10]))
print("X_test dtype:", getattr(X_train, "dtype", type(X_train)))
print("X_test sample:", np.array(X_test[:10]))

# Baseline model to see initial performance
linear_pipeline.fit(X_train, np.ravel(y_train))

# Predicting on X_train sees if model is overfitting
train_pred = linear_pipeline.predict(X_train)
test_pred = linear_pipeline.predict(X_test)


# The plot is too perfect. Could be data leakage
plt.figure(figsize=(6,6))
plt.scatter(y_train, train_pred, color='blue', label='Train')
plt.scatter(y_test, test_pred, color='red', label='Test')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual')
plt.legend()
plt.show()

# Transform it back to original units since it was transformed
original_y_test = np.exp(test_pred)

# mean squared error is very low. could be overfitting
print('Mean Squared Error: ', mean_squared_error(y_test, original_y_test))
print('R^2: ', r2_score(y_test, original_y_test))

# Advanced model with hyperparameter tuning also combines preprocessing
lgbm_pipeline = Pipeline(steps=[
    ('preprocessing', preprocess),
    ('lgbm_model', lgbm)
])

# Parameters still are not helping the model
grid = {
    'lgbm_model__max_depth' : [1, 2, 3, 4],
    'lgbm_model__num_leaves' : [2, 4, 5],
    'lgbm_model__min_data_in_leaf' : [2, 3, 4, 5],
    'lgbm_model__n_estimators' : [50, 100]
}

lgbm_pipeline.fit(X_train, np.ravel(y_train))

lgbm_pred = lgbm_pipeline.predict(X_test)
#lgbm_pred = np.argmax(lgbm_pred, axis=1)
print(lgbm_pred)
print('Mean Squared Error:', mean_squared_error(y_test, lgbm_pred))

#early_stopping = lgb.early_stopping(stopping_rounds=50)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lgbm_pipeline, X, y, cv=kf)

search = RandomizedSearchCV(estimator=lgbm_pipeline, param_distributions=grid, 
                           n_iter=20, cv=kf, scoring='r2', n_jobs=-1,
                            verbose=2, random_state=42, refit=True)

#search.fit(X_train, np.ravel(y_train), lgbm_model__callbacks=[early_stopping])
search.fit(X_train, np.ravel(y_train))


#https://chatgpt.com/c/68f1505c-21d0-8331-8f98-fd05007cb3f6
#https://chatgpt.com/c/68f2a0a3-3a7c-832b-babb-839d05a7c568
#https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

#https://gemini.google.com/u/2/app/62dab08f8df5ad32?pageId=none
#https://www.google.com/search?q=why+is+my+ml+model+with+randomizedsearchcv+not+improving+with+a+parameter+grid&sca_esv=c2292d1d59604bd8&sxsrf=AE3TifN5jDc39h-Bwpr3gAhfGlGCRTYj8A%3A1761867985855&ei=0fgDae74M8-r5NoPqIrsmQ0&ved=0ahUKEwiupcDbjc2QAxXPFVkFHSgFO9MQ4dUDCBM&uact=5&oq=why+is+my+ml+model+with+randomizedsearchcv+not+improving+with+a+parameter+grid&gs_lp=Egxnd3Mtd2l6LXNlcnAiTndoeSBpcyBteSBtbCBtb2RlbCB3aXRoIHJhbmRvbWl6ZWRzZWFyY2hjdiBub3QgaW1wcm92aW5nIHdpdGggYSBwYXJhbWV0ZXIgZ3JpZEjNoAFQjAhYh54BcA14AZABAJgBqwGgAaZBqgEFMzIuNDe4AQPIAQD4AQGYAk2gAoU-qAIQwgIHECMYJxjqAsICChAjGPAFGCcY6gLCAhQQABiABBiRAhi0AhiKBRjqAtgBAcICBBAjGCfCAgoQIxiABBgnGIoFwgIKEAAYgAQYQxiKBcICCxAAGIAEGLEDGIMBwgILEC4YgAQYsQMYgwHCAhEQLhiABBixAxjRAxiDARjHAcICDhAAGIAEGLEDGIMBGIoFwgIOEC4YgAQYsQMY0QMYxwHCAggQLhiABBixA8ICBRAAGIAEwgIIEAAYgAQYsQPCAgQQABgDwgIQECMY8AUYgAQYJxjJAhiKBcICCxAAGIAEGJECGIoFwgIHEAAYgAQYCsICCxAAGIAEGIYDGIoFwgIFEAAY7wXCAgYQABgWGB7CAggQABiiBBiJBcICBRAhGKABwgIFECEYnwXCAgUQIRirAsICCBAAGIAEGKIEwgIHECEYoAEYCpgDCPEFozIDnJYftea6BgYIARABGAGSBwUyMy41NKAH76IDsgcFMTQuNTS4B8M9wgcJMC4yNy40OS4xyAeoAg&sclient=gws-wiz-serp
#https://gemini.google.com/app/15b3248b3644369f
