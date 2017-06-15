import numpy as np
import bigfloat as bg
import math
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.cluster import KMeans


def generate_timeseries(prices, time_length):
    
    if(len(prices) < time_length):
        print "Not enough data!"
        return 0
    
    m = len(prices) - time_length
    timeseries = np.empty((m, time_length+1))
    
    for i in range(m):
        timeseries[i, :time_length] = prices[i:i+time_length]
        timeseries[i, time_length] = prices[i + time_length] - prices[i + time_length - 1]
    
    
    return timeseries

def clusters(ts, k):
    return KMeans(n_clusters=k).fit(ts).cluster_centers_

def choose_effective_centers(centers, n):
    return centers[np.argsort(np.ptp(centers, axis=1))[-n:]]

def euclidean_distance(x,y):
 
    return math.sqrt(sum(math.pow(a-b,2) for a, b in zip(x, y)))


def similarity(a, b):
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a, dtype = np.float64)
    std_b = np.std(b, dtype = np.float64)
    return sum((a - mean_a) * (b - mean_b)) / (len(a) * std_a * std_b)


def predict_dpi(x, s):
    
    num = 0
    den = 0
    for i in range(len(s)):
        y_i = s[i, len(x)]
        x_i = s[i, :len(x)]
        ex = bg.exp(-0.25*(math.pow(euclidean_distance(x,x_i),2)))
        num = bg.add(num,bg.mul(y_i,ex))
        den = bg.add(den,ex)
    return bg.div(num,den)



def linear_regression_vars(prices, v_bid, v_ask, s1, s2, s3):
    
    X = np.empty((len(prices) - 121, 4))
    Y = np.empty(len(prices) - 121)
    for i in range(120, len(prices) - 1):
        dp = prices[i + 1] - prices[i]
        dp1 = predict_dpi(prices[i - 30:i], s1)
        dp2 = predict_dpi(prices[i - 60:i], s2)
        dp3 = predict_dpi(prices[i - 120:i], s3)
        r = (v_bid[i] - v_ask[i]) / (v_bid[i] + v_ask[i])
        X[i - 120, :] = [dp1, dp2, dp3, r]
        Y[i - 120] = dp
    return X, Y

def find_parameters_w(X, Y):

    clf = linear_model.LinearRegression()
    clf.fit(X, Y)
    w0 = clf.intercept_
    w1, w2, w3, w4 = clf.coef_
    return w0, w1, w2, w3, w4

def predict_dps(prices, v_bid, v_ask, s1, s2, s3, w):
    
    dps = []
    w0, w1, w2, w3, w4 = w
    for i in range(120, len(prices) - 1):
        dp1 = predict_dpi(prices[i - 30:i], s1)
        dp2 = predict_dpi(prices[i - 60:i], s2)
        dp3 = predict_dpi(prices[i - 120:i], s3)
        r = (v_bid[i] - v_ask[i]) / (v_bid[i] + v_ask[i])
        dp = bg.add(w0,bg.add(bg.mul(w1,dp1),bg.add(bg.mul(w2,dp2),bg.add(bg.mul(w3,dp3),bg.mul(w4,r)))))
        dps.append(float(dp))
    return dps

def evaluate_performance(prices, dps, t, step):
    
    bank_balance = 0
    position = 0
    for i in range(120, len(prices) - 1, step):
        # long position - BUY
        if dps[i - 120] > t and position <= 0:
            position += 1
            bank_balance -= prices[i]
        # short position - SELL
        if dps[i - 120] < -t and position >= 0:
            position -= 1
            bank_balance += prices[i]
            
        print bank_balance
    # sell what you bought
    if position == 1:
        bank_balance += prices[len(prices) - 1]
    # pay back what you borrowed
    if position == -1:
        bank_balance -= prices[len(prices) - 1]
    return bank_balance





