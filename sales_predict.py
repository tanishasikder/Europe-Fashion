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
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

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

channel = pd.DataFrame(channel_data)
types = pd.DataFrame(types_data)

pivot = pivot.dropna()
pivot = pivot.drop(19)
pivot = pivot.rename(columns={'Unnamed: 0' : 'Row Labels', 'Unnamed: 1' : 'Dresses', 'Unnamed: 2' : 'Pants', 
                      'Unnamed: 3' : 'Shoes','Unnamed: 4' : 'Sleepwear', 'Unnamed: 5' : 'T-Shirts', 
                      'Unnamed: 6' : 'Grand Total'})

