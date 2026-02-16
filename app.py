import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from agent.agent import Agent, load_model
from functions import *
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="RL Stock Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Zerodha-inspired Theme
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fb;
        color: #212529;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #387ed1 !important;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e3e9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 3px;
        height: 2.8em;
        background-color: #387ed1;
        color: white;
        border: none;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        color: #387ed1;
    }
    /* Customize sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e3e9;
    }
</style>
""", unsafe_allow_html=True)

# Unified Dashboard Heading
st.title("RL Based - Stock Trading System")
st.markdown("""
<div style='background-color: #387ed1; height: 2px; margin-bottom: 20px;'></div>
""", unsafe_allow_html=True)

# Sidebar for Configuration
st.sidebar.header("⚙️ Market Watch")

# Get available data files
data_dir = "data/"
available_stocks = [f.replace(".csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")]
selected_stock = st.sidebar.selectbox("Select Stock/Instrument", available_stocks)

# Get available models
models_dir = "models/"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Support both .pth (PyTorch) and .pkl (legacy/backup)
available_models = [m for m in os.listdir(models_dir) if m.endswith(".pth") or m.endswith(".pkl")]

if not available_models:
    st.sidebar.warning("⚠️ No trained models found. Please train first.")
    selected_model = "None"
else:
    selected_model = st.sidebar.selectbox("Select Strategy Model", ["None"] + available_models)
    
st.sidebar.markdown("---")
fast_mode = st.sidebar.checkbox("Fast Performance Mode", value=True, help="Optimizes for speed by reducing UI update overhead.")
window_size = st.sidebar.slider("Historical Window", min_value=5, max_value=50, value=10)
episode_count = st.sidebar.number_input("Training Episodes", min_value=1, max_value=5000, value=10)
st.sidebar.markdown("---")

# Main Dashboard Layout
top_m1, top_m2, top_m3, top_m4 = st.columns(4)
overall_pl = top_m1.empty()
overall_accuracy = top_m2.empty()
total_trades = top_m3.empty()
market_status = top_m4.empty()

# Initialization of placeholders
overall_pl.metric("Total P&L", "$0.00", "0.00%")
overall_accuracy.metric("Accuracy", "0.00%")
total_trades.metric("Total Trades", "0")
market_status.metric("Market Status", "Idle", delta_color="off")

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📈 Strategy Training")
    start_training = st.button("Initialize & Start Training", key="train_btn")
    train_progress = st.progress(0)
    train_status = st.empty()
    train_chart = st.empty()
    train_metrics = st.empty()

with col_right:
    st.subheader("📊 Strategy Evaluation")
    run_evaluation = st.button("Run Performance Scan", key="eval_btn")
    eval_chart = st.empty()
    eval_metrics_top = st.empty()
    eval_metrics_bottom = st.empty()

# --- Shared Metrics Display Function ---
def display_zerodha_metrics(metrics_placeholder, p_l, accuracy, buy_count, sell_count, total_profit, total_loss):
    with metrics_placeholder.container():
        st.markdown(f"""
        <div class="metric-card">
            <h4 style='margin-top:0; color:#666; font-size:0.9em;'>TRADING PERFORMANCE</h4>
            <table style='width:100%; border-collapse: collapse;'>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding:8px 0;'>Net P&L</td>
                    <td style='text-align:right; font-weight:bold; color:{"#4caf50" if p_l >= 0 else "#f44336"}'>{formatPrice(p_l)}</td>
                </tr>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding:8px 0;'>Accuracy</td>
                    <td style='text-align:right;'>{accuracy:.2%}</td>
                </tr>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding:8px 0;'>Total Trades (B/S)</td>
                    <td style='text-align:right;'>{buy_count} / {sell_count}</td>
                </tr>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding:8px 0;'>Gross Profit</td>
                    <td style='text-align:right; color:#4caf50;'>{formatPrice(total_profit)}</td>
                </tr>
                <tr>
                    <td style='padding:8px 0;'>Gross Loss</td>
                    <td style='text-align:right; color:#f44336;'>{formatPrice(total_loss)}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# --- Modular Evaluation Function ---
def run_evaluation_logic(agent, stock_key, window_size_val, chart_placeholder, metrics_top_placeholder, metrics_bottom_placeholder, is_live=False):
    try:
        # Preserve training state
        prev_is_eval = agent.is_eval
        prev_inventory = agent.inventory
        agent.is_eval = True
        agent.inventory = []

        data = getStockDataVec(stock_key)
        l = len(data) - 1
        
        state = getState(data, 0, window_size_val + 1)
        total_p_l = 0
        sell_count_ev = 0
        buy_count_ev = 0
        profitable_sells_ev = 0
        gross_profit_ev = 0
        gross_loss_ev = 0
        actions = []
        
        for t in range(l):
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size_val + 1)
            
            if action == 1: # buy
                agent.inventory.append(data[t])
                actions.append({'Type': 'BUY', 'Time': t, 'Price': data[t], 'P&L': 0.0})
                buy_count_ev += 1
            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = agent.inventory.pop(0)
                profit = data[t] - bought_price
                total_p_l += profit
                sell_count_ev += 1
                if profit > 0:
                    profitable_sells_ev += 1
                    gross_profit_ev += profit
                else:
                    gross_loss_ev += abs(profit)
                actions.append({'Type': 'SELL', 'Time': t, 'Price': data[t], 'P&L': profit})
            
            state = next_state
        
        accuracy_ev = (profitable_sells_ev / sell_count_ev) if sell_count_ev > 0 else 0.0
        
        # Restore training state
        agent.is_eval = prev_is_eval
        agent.inventory = prev_inventory
        
        # Update metrics and charts
        display_zerodha_metrics(metrics_top_placeholder, total_p_l, accuracy_ev, buy_count_ev, sell_count_ev, gross_profit_ev, gross_loss_ev)
        
        if not is_live:
            overall_pl.metric("Total P&L (Scan)", formatPrice(total_p_l), f"{(total_p_l/data[0]):.2%}")
            overall_accuracy.metric("Accuracy (Scan)", f"{accuracy_ev:.2%}")
            total_trades.metric("Total Trades (Scan)", str(buy_count_ev + sell_count_ev))

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data, color='#999', alpha=0.3, label="Market Price", linewidth=1)
        
        buys = [a for a in actions if a['Type'] == 'BUY']
        sells = [a for a in actions if a['Type'] == 'SELL']
        
        if buys:
            ax.scatter([b['Time'] for b in buys], [b['Price'] for b in buys], color='#4caf50', marker='^', s=45, label='Buy Entry', zorder=5)
        if sells:
            ax.scatter([s['Time'] for s in sells], [s['Price'] for s in sells], color='#f44336', marker='v', s=45, label='Sell Exit', zorder=5)
        
        ax.set_title(f"Strategic Market Scan: {stock_key} ({'LATEST' if is_live else 'HISTORICAL'})", color='#387ed1', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.set_facecolor('#fdfdfd')
        fig.patch.set_facecolor('#f8f9fb')
        ax.legend(loc='best', fontsize=8)
        chart_placeholder.pyplot(fig)
        plt.close(fig)

        if actions and not is_live:
            with metrics_bottom_placeholder.container():
                st.markdown("### 📝 Trade Book")
                df_actions = pd.DataFrame(actions)
                df_actions = df_actions.tail(100)[::-1]
                def highlight_trade(row):
                    if row['Type'] == 'BUY': return ['background-color: rgba(76, 175, 80, 0.05)'] * 4
                    if row['Type'] == 'SELL': 
                        color = 'rgba(76, 175, 80, 0.15)' if row['P&L'] > 0 else 'rgba(244, 67, 54, 0.15)'
                        return [f'background-color: {color}'] * 4
                    return [''] * 4
                st.dataframe(df_actions.style.apply(highlight_trade, axis=1).format({'Price': '${:.2f}', 'P&L': '${:.2f}'}), height=300, use_container_width=True)

    except Exception as e:
        st.error(f"Strategy Scan Error: {str(e)}")

# --- Training Logic ---
if start_training:
    market_status.metric("Market Status", "Training...", delta="LIVE", delta_color="normal")
    agent = Agent(window_size)
    data = getStockDataVec(selected_stock)
    l = len(data) - 1
    batch_size = 32
    
    all_profits = []
    all_accuracies = []
    
    for e in range(episode_count + 1):
        train_status.text(f"Episode {e}/{episode_count}")
        train_progress.progress(e / episode_count if episode_count > 0 else 1.0)
        
        state = getState(data, 0, window_size + 1)
        total_profit_ep = 0
        agent.inventory = []
        sell_count_ep = 0
        profitable_sells_ep = 0
        gross_profit_ep = 0
        gross_loss_ep = 0
        buy_count_ep = 0

        for t in range(l):
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1: # buy
                agent.inventory.append(data[t])
                buy_count_ep += 1
            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = agent.inventory.pop(0)
                profit = data[t] - bought_price
                reward = max(profit, 0)
                total_profit_ep += profit
                sell_count_ep += 1
                if profit > 0:
                    profitable_sells_ep += 1
                    gross_profit_ep += profit
                else:
                    gross_loss_ep += abs(profit)

            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                all_profits.append(total_profit_ep)
                accuracy_ep = (profitable_sells_ep / sell_count_ep) if sell_count_ep > 0 else 0.0
                all_accuracies.append(accuracy_ep)
                
                # Update Metrics
                display_zerodha_metrics(train_metrics, total_profit_ep, accuracy_ep, buy_count_ep, sell_count_ep, gross_profit_ep, gross_loss_ep)
                
                # Update Top Row
                overall_pl.metric("Total P&L", formatPrice(np.mean(all_profits)), f"{np.mean(all_profits)/data[0]:.2%}")
                overall_accuracy.metric("Accuracy", f"{np.mean(all_accuracies):.2%}")
                total_trades.metric("Total Trades", str(len(all_profits) * (buy_count_ep + sell_count_ep)))

                # Update Chart
                if not fast_mode or e % 2 == 0 or e == episode_count:
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    ax1.plot(all_profits, color='#387ed1', label='Batch Profit', linewidth=2)
                    ax1.set_xlabel("Episode")
                    ax1.set_ylabel("Profit ($)", color='#387ed1')
                    ax1.tick_params(axis='y', labelcolor='#387ed1')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(all_accuracies, color='#ff9800', label='Accuracy', linestyle='--', alpha=0.7)
                    ax2.set_ylabel("Win Rate", color='#ff9800')
                    ax2.tick_params(axis='y', labelcolor='#ff9800')
                    
                    fig.tight_layout()
                    ax1.set_title("Training Performance (Vectorized)", color='#387ed1', fontsize=12)
                    ax1.grid(True, linestyle='--', alpha=0.3)
                    train_chart.pyplot(fig)
                    plt.close(fig)
                    
                    # RUN SIMULTANEOUS EVALUATION
                    run_evaluation_logic(agent, selected_stock, window_size, eval_chart, eval_metrics_top, eval_metrics_bottom, is_live=True)

            # Call expReplay every 4 steps in fast mode
            train_interval = 4 if fast_mode else 1
            if len(agent.memory) > batch_size and t % train_interval == 0:
                agent.expReplay(batch_size)

        if e % 10 == 0:
            agent.save_model(f"models/model_ep{e}.pth")
    
    train_status.text("✅ Training Cycle Complete")
    market_status.metric("Market Status", "Complete", delta_color="off")
    st.success("Strategy training completed successfully!")

# --- Evaluation Logic ---
if run_evaluation:
    if selected_model == "None":
        st.warning("Please select a strategy model from the sidebar.")
    else:
        market_status.metric("Market Status", "Evaluating...", delta="SCAN", delta_color="normal")
        try:
            model_path = "models/" + selected_model
            # Open the state dict to verify window size (feature count)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            # Detect actual window size from model architecture (fc1.weight shape)
            model_window_size = state_dict['fc1.weight'].shape[1]
            
            if model_window_size != window_size:
                st.info(f"ℹ️ Model trained with window size {model_window_size}. Overriding sidebar setting ({window_size}) for accuracy.")
            
            agent = Agent(model_window_size, is_eval=True, model_name=selected_model) 
            run_evaluation_logic(agent, selected_stock, model_window_size, eval_chart, eval_metrics_top, eval_metrics_bottom, is_live=False)
            market_status.metric("Market Status", "Scan Ready", delta_color="off")
        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            market_status.metric("Market Status", "Error", delta_color="off")
