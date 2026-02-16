import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent.agent import Agent
from functions import *

if len(sys.argv) != 4:
    print("Usage: python run_plot.py [stock] [window] [episodes]")
    exit()

stock_name = sys.argv[1]
window_size = int(sys.argv[2])
episode_count = int(sys.argv[3])

data = getStockDataVec(stock_name)
l = len(data) - 1

agent = Agent(window_size)
batch_size = 32

profits = []
last_episode_trades = []

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0.0
    agent.inventory = []
    trades = []  # (t, action, price)

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            agent.inventory.append(data[t])
            trades.append((t, 'buy', data[t]))

        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            profit = data[t] - bought_price
            reward = max(profit, 0)
            total_profit += profit
            trades.append((t, 'sell', data[t]))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    print("Total Profit: " + formatPrice(total_profit))
    profits.append(total_profit)
    last_episode_trades = trades

    if e % 10 == 0:
        # save model snapshot (pickle) if agent supports it
        try:
            agent.model.save(os.path.join('models', 'model_ep' + str(e)))
        except Exception:
            pass

# Ensure images dir
os.makedirs('images', exist_ok=True)

# Plot profits per episode
plt.figure(figsize=(8,4))
plt.bar(range(len(profits)), profits, color='tab:blue')
plt.xlabel('Episode')
plt.ylabel('Total Profit')
plt.title(f'Profits per Episode ({stock_name})')
profits_path = os.path.join('images', 'profits_per_episode.png')
plt.tight_layout()
plt.savefig(profits_path)
plt.close()

# Plot price series with last episode trades
plt.figure(figsize=(12,5))
plt.plot(data, label='Price')
buys = [ (t, p) for (t,a,p) in last_episode_trades if a=='buy' ]
sells = [ (t, p) for (t,a,p) in last_episode_trades if a=='sell' ]
if buys:
    plt.scatter([t for t,p in buys], [p for t,p in buys], marker='^', color='g', label='Buy')
if sells:
    plt.scatter([t for t,p in sells], [p for t,p in sells], marker='v', color='r', label='Sell')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Price and Trades (last episode) - {stock_name}')
plt.legend()
trades_path = os.path.join('images', 'trades_last_episode.png')
plt.tight_layout()
plt.savefig(trades_path)
plt.close()

print('\nSaved graphs:')
print(' -', profits_path)
print(' -', trades_path)
