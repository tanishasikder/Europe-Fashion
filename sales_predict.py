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
'''