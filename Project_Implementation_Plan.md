# Unified Strategy Dashboard Enhancement

This plan outlines the integration of training and evaluation into a single execution flow, ensuring metrics are synced and errors are minimized.

## User Review Required

> [!IMPORTANT]
> - **Unified Run**: Training will now automatically trigger an evaluation periodically to show real-time performance on market data.
> - **Input Sync**: The historical window size from the sidebar will be the primary source of truth for both training and evaluation.

## Proposed Changes

### Dashboard Core [app.py]

#### [MODIFY] [app.py](file:///c:/Users/eshas/Downloads/RL/q-trader-master/app.py)
- **Modularize Evaluation**: Extract evaluation logic into a reusable `run_strategy_eval` function.
- **Periodic Evaluation**: Call `run_strategy_eval` inside the training loop (e.g., every few episodes) to provide "simultaneous" updates.
- **Remove Size Check**: Delete the arbitrary 1000-byte model size check.
- **Window Sync**: Ensure `window_size` slider is consistently used for state representation.

## Verification Plan

### Automated Tests
- Run `python -m py_compile app.py` to ensure no syntax errors.

### Manual Verification
- Start training and verify that the "Strategy Evaluation" chart updates automatically.
- Run a manual "Performance Scan" and verify accuracy and trade history.
- Change the "Historical Window" slider and ensure it affects both training and evaluation states.
- **Detailed History**: Zerodha-style trade history table with color-coded profit/loss per trade.
- **Values**: Gross Profit, Gross Loss, Net P&L, Accuracy, Buy Count, Sell Count.

## Verification Plan

1. Launch app: `python -m streamlit run app.py`
2. Verify all components are visible on a single page.
3. Test training and verify that metrics update in real-time.
4. Test evaluation and verify the trade history matches the visualization.

