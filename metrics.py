from agent.agent import Agent, load_model
from functions import *
import sys

if len(sys.argv) != 3:
    print("Usage: python metrics.py [stock] [model]")
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]

model = load_model("models/" + model_name)
window_size = model.input_shape[1]

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1

state = getState(data, 0, window_size + 1)
total_profit = 0.0
agent.inventory = []

sell_count = 0
profitable_sells = 0

for t in range(l):
    action = agent.act(state)

    next_state = getState(data, t + 1, window_size + 1)
    reward = 0

    if action == 1:  # buy
        agent.inventory.append(data[t])

    elif action == 2 and len(agent.inventory) > 0:  # sell
        bought_price = agent.inventory.pop(0)
        profit = data[t] - bought_price
        reward = max(profit, 0)
        total_profit += profit
        sell_count += 1
        if profit > 0:
            profitable_sells += 1

    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

accuracy = (profitable_sells / sell_count) if sell_count > 0 else 0.0

print(f"Model: {model_name}")
print(f"Stock: {stock_name}")
print(f"Total sells: {sell_count}")
print(f"Profitable sells: {profitable_sells}")
print(f"Accuracy (profitable sells / total sells): {accuracy:.4f}")
print("Total Profit: " + formatPrice(total_profit))
