from pymongo import MongoClient
from bayesian_regression import *

client = MongoClient()
database = client['mercado_bitcoin']
collection = database['data']

# Retrieve price, v_ask, and v_bid data points from the database.
prices = []
v_ask = []
v_bid = []
num_points = 777600
for doc in collection.find().limit(num_points):
    prices.append(doc['price'])
    v_ask.append(doc['v_ask'])
    v_bid.append(doc['v_bid'])

# Divide prices into three, roughly equal sized, periods:
# prices1, prices2, and prices3.
[prices1, prices2, prices3] = np.array_split(prices, 3)

# Divide v_bid into three, roughly equal sized, periods:
# v_bid1, v_bid2, and v_bid3.
[v_bid1, v_bid2, v_bid3] = np.array_split(v_bid, 3)

# Divide v_ask into three, roughly equal sized, periods:
# v_ask1, v_ask2, and v_ask3.
[v_ask1, v_ask2, v_ask3] = np.array_split(v_ask, 3)

# Use the first time period (prices1) to generate all possible time series of
# appropriate length (180, 360, and 720).
timeseries30 = generate_timeseries(prices1, 30)
timeseries60 = generate_timeseries(prices1, 60)
timeseries120 = generate_timeseries(prices1, 120)

clusters30 = clusters(timeseries30,100)
clusters60 = clusters(timeseries60,100)
clusters120 = clusters(timeseries120,100)

s1 = choose_effective_centers(clusters30,20)
s2 = choose_effective_centers(clusters60,20)
s3 = choose_effective_centers(clusters120,20)

X,Y = linear_regression_vars(prices2, v_bid2, v_ask2, s1, s2, s3)

w = find_parameters_w(X,Y)

dps = predict_dps(prices3, v_bid3, v_ask3, s1, s2, s3, w)

bank_balance = evaluate_performance(prices3, dps, t=0.0001, step=1)

print "FINAL BANK BALANCE :    "
print bank_balance