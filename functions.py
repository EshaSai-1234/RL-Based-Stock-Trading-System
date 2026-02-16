import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	# compute log returns for the block
	res = []
	eps = 1e-8
	log_returns = []
	for i in range(n - 1):
		a = max(block[i], eps)
		b = max(block[i+1], eps)
		log_returns.append(math.log(b / a))

	lr_std = (np.std(log_returns) + 1e-6)
	sma = float(np.mean(block))

	for i in range(n - 1):
		lr_norm = log_returns[i] / lr_std
		sma_diff = (block[i+1] - sma) / (sma + eps)
		# combine normalized log-return with SMA deviation and pass through sigmoid
		val = sigmoid(lr_norm + sma_diff)
		res.append(val)

	return np.array([res])
