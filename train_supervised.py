import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from functions import getStockDataVec, getState

def build_dataset(data, window_size):
    X = []
    y = []
    l = len(data) - 1
    for t in range(l - 1):
        # state ending at t (uses window_size+1 in original code pattern)
        s = getState(data, t, window_size + 1)
        # flatten
        X.append(s.flatten())
        # label = 1 if next day's price increased, else 0
        y.append(1 if data[t+1] > data[t] else 0)

    return np.array(X), np.array(y)

def main():
    stock = "^GSPC_2011"
    window = 10

    data = getStockDataVec(stock)
    X, y = build_dataset(data, window)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, os.path.join('models', 'rf_direction.pkl'))

    print(f"Direction accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, preds))

if __name__ == '__main__':
    main()
