import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt

# Load Data
stock_data = yf.download('AAPL', start='2020-01-01', end='2022-01-01', auto_adjust=True)

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

# FEATURE ENGINEERING

# Base Feature: Log Returns
stock_data['Log_Ret'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))

# --- Original Indicators ---
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['Dist_SMA_20'] = (stock_data['Close'] - stock_data['SMA_20']) / stock_data['SMA_20']
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['Dist_SMA_50'] = (stock_data['Close'] - stock_data['SMA_50']) / stock_data['SMA_50']
stock_data['Volatility'] = stock_data['Log_Ret'].rolling(window=20).std()

# 10 NEW INDICATORS

# RSI
delta = stock_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# MACD (Normalized)
ema_12 = stock_data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
stock_data['MACD'] = (ema_12 - ema_26) / stock_data['Close']

# MACD Signal
stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Width
bb_std = stock_data['Close'].rolling(window=20).std()
stock_data['BB_Width'] = (4 * bb_std) / stock_data['SMA_20']

# ROC
stock_data['ROC_10'] = stock_data['Close'].pct_change(periods=10)

# ATR (Normalized)
prev_close = stock_data['Close'].shift(1)
tr1 = stock_data['High'] - stock_data['Low']
tr2 = (stock_data['High'] - prev_close).abs()
tr3 = (stock_data['Low'] - prev_close).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
stock_data['ATR_Norm'] = tr.rolling(window=14).mean() / stock_data['Close']

# CCI
tp = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
tp_sma = tp.rolling(window=20).mean()
md = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
stock_data['CCI'] = (tp - tp_sma) / (0.015 * md)

# Williams %R
hh_14 = stock_data['High'].rolling(window=14).max()
ll_14 = stock_data['Low'].rolling(window=14).min()
stock_data['Williams_R'] = -100 * ((hh_14 - stock_data['Close']) / (hh_14 - ll_14))

# Volume Change
stock_data['Vol_Change'] = stock_data['Volume'].pct_change()

# High-Low Range
stock_data['HL_Pct'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close']


# Create Target
stock_data['Target'] = stock_data['Log_Ret'].shift(-1)
stock_data.dropna(inplace=True)

# Inputs
feature_cols = [
    'Log_Ret', 'Dist_SMA_20', 'Dist_SMA_50', 'Volatility',
    'RSI', 'MACD', 'MACD_Signal', 'BB_Width', 'ROC_10',
    'ATR_Norm', 'CCI', 'Williams_R', 'Vol_Change', 'HL_Pct']

X_raw = stock_data[feature_cols]
y = stock_data['Target']

# Split
split_index = int(len(stock_data) * 0.8)
X_train_raw = X_raw.iloc[:split_index]
X_test_raw = X_raw.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Explained Variance Plot
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 4))
plt.bar(range(len(explained_variance)), explained_variance)
plt.title('Explained Variance (5 Components)')
plt.show()


# MODEL TRAINING
# Model 1: PCA
model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
mse_pca = mean_squared_error(y_test, y_pred_pca)

# Model 2: No PCA
model_no_pca = LinearRegression()
model_no_pca.fit(X_train_scaled, y_train)
y_pred_no_pca = model_no_pca.predict(X_test_scaled)
mse_no_pca = mean_squared_error(y_test, y_pred_no_pca)

print(f'MSE with PCA: {mse_pca}')
print(f'MSE without PCA: {mse_no_pca}')


# VISUALIZATION: Predicted vs Actual Returns

# Generate predictions on Training data to show fit
y_train_pred_pca = model_pca.predict(X_train_pca)
y_train_pred_no_pca = model_no_pca.predict(X_train_scaled)

# PLOT 1: Test Data Prediction (Zoomed in on last 50 days for clarity)
plt.figure(figsize=(14, 8))

# Zooming in on the last 50 days of the test set makes it easier to see the difference
zoom = 50

plt.subplot(2, 1, 1)
plt.plot(y_test.index[-zoom:], y_test[-zoom:], label='Actual Return', color='blue', marker='o', alpha=0.6)
plt.plot(y_test.index[-zoom:], y_pred_pca[-zoom:], label='Predicted (PCA)', color='red', marker='x', linestyle='--')
plt.title(f'Test Data (Last 50 Days): PCA Model (MSE: {mse_pca:.6f})')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(y_test.index[-zoom:], y_test[-zoom:], label='Actual Return', color='green', marker='o', alpha=0.6)
plt.plot(y_test.index[-zoom:], y_pred_no_pca[-zoom:], label='Predicted (No PCA)', color='orange', marker='x', linestyle='--')
plt.title(f'Test Data (Last 50 Days): No-PCA Model (MSE: {mse_no_pca:.6f})')
plt.legend()

plt.tight_layout()
plt.show()

# STRATEGY PERFORMANCE, Buy signal when predicted return is positive

actual_returns = np.exp(y_test.values) - 1

signal_pca = np.where(y_pred_pca > 0, 1, 0)
strategy_returns_pca = signal_pca * actual_returns

signal_no_pca = np.where(y_pred_no_pca > 0, 1, 0)
strategy_returns_no_pca = signal_no_pca * actual_returns

cumulative_returns_pca = (1 + strategy_returns_pca).cumprod()
cumulative_returns_no_pca = (1 + strategy_returns_no_pca).cumprod()
cumulative_returns_bh = (1 + actual_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, cumulative_returns_bh, label='Baseline (Buy & Hold)', color='gray', linestyle='--', linewidth=2)
plt.plot(y_test.index, cumulative_returns_pca, label='PCA Model Strategy', color='blue', linewidth=2)
plt.plot(y_test.index, cumulative_returns_no_pca, label='No-PCA Model Strategy', color='green', linewidth=2)

plt.title('Cumulative Returns: PCA Strategy vs No-PCA Strategy vs Baseline')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True)
plt.show()