import matplotlib.pyplot as plt
import pandas as pd
import os

# Ensure images directory exists
os.makedirs('images', exist_ok=True)

# Load data
data_path = 'data/^GSPC_2011.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    prices = df['Close'].tolist()
    
    plt.figure(figsize=(10, 6))
    plt.plot(prices, label='S&P 500 (^GSPC_2011)')
    plt.title('Stock Price - Evaluation Period')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plot_path = 'images/trade_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
else:
    print(f"Data file {data_path} not found.")
