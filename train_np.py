import numpy as np
from functions import getStockDataVec, getState

def build_dataset(data, window_size):
    X = []
    y = []
    l = len(data) - 1
    for t in range(l - 1):
        s = getState(data, t, window_size + 1)
        X.append(s.flatten())
        y.append(1 if data[t+1] > data[t] else 0)
    return np.array(X), np.array(y)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, lr=0.01, epochs=500, reg=1e-4):
    n, m = X.shape
    # add bias
    Xb = np.hstack([np.ones((n,1)), X])
    w = np.zeros(Xb.shape[1])

    for ep in range(epochs):
        z = Xb.dot(w)
        preds = sigmoid(z)
        error = preds - y
        grad = (Xb.T.dot(error)) / n + reg * np.r_[0, w[1:]]
        w -= lr * grad
        if ep % 100 == 0:
            loss = -np.mean(y * np.log(preds + 1e-9) + (1-y) * np.log(1 - preds + 1e-9))
            pass
    return w

def evaluate(w, X, y):
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    preds = sigmoid(Xb.dot(w))
    labels = (preds >= 0.5).astype(int)
    acc = (labels == y).mean()
    tp = int(((labels==1) & (y==1)).sum())
    tn = int(((labels==0) & (y==0)).sum())
    fp = int(((labels==1) & (y==0)).sum())
    fn = int(((labels==0) & (y==1)).sum())
    return acc, tp, tn, fp, fn

def main():
    stock = "^GSPC_2011"
    window = 10
    data = getStockDataVec(stock)
    X, y = build_dataset(data, window)
    # simple shuffle and split
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    w = train_logistic(Xtr, ytr, lr=0.05, epochs=1000, reg=1e-4)
    acc, tp, tn, fp, fn = evaluate(w, Xte, yte)

    print(f"Logistic accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

if __name__ == '__main__':
    main()
