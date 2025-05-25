"""
机器学习模型模块

lstm模型预测收益率 py3.9
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


def _load_tickers(csv_path):
    df = pd.read_csv(csv_path)
    return df['ticker'].unique().tolist()


def _get_stock_data(ticker):
    data = yf.download(ticker, period='5y')['Close']
    return np.log(data / data.shift(1)).dropna()


class StockDataset(Dataset):
    def __init__(self, sequences_5d, sequences_10d, window_size):
        self.sequences_5d = sequences_5d
        self.sequences_10d = sequences_10d
        self.window_size = window_size

    def __len__(self):
        return len(self.sequences_5d)

    def __getitem__(self, idx):
        seq = self.sequences_5d[idx][0]
        label_5d = self.sequences_5d[idx][1]
        label_10d = self.sequences_10d[idx][1]
        return (torch.as_tensor(seq, dtype=torch.float32).view(self.window_size, 1),
                torch.as_tensor([label_5d], dtype=torch.float32),
                torch.as_tensor([label_10d], dtype=torch.float32))


def save_results(df, output_path):
    df.to_csv(output_path, index=False)


class LSTMPredictor:
    def __init__(self, tickers, window_size=30):
        self.tickers = tickers
        self.window_size = window_size
        self.models_5d = {}
        self.models_10d = {}
        self.scalers = {}

    def _create_sequences(self, data):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data.values.reshape(-1, 1))

        sequences_5d = []
        sequences_10d = []
        for i in range(len(scaled) - self.window_size - 10):
            seq = scaled[i:i + self.window_size].flatten()
            label_5d = scaled[i + self.window_size:i + self.window_size + 5].mean()
            label_10d = scaled[i + self.window_size:i + self.window_size + 10].mean()
            sequences_5d.append((seq, label_5d))
            sequences_10d.append((seq, label_10d))
        return sequences_5d, sequences_10d, scaler

    class LSTMModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(0)
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    def train(self, csv_path, epochs=50):
        self.tickers = _load_tickers(csv_path)
        for ticker in self.tickers:
            data = _get_stock_data(ticker)
            seq_5d, seq_10d, scaler = self._create_sequences(data)

            dataset = StockDataset(seq_5d, seq_10d, self.window_size)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            model_5d = self.LSTMModel()
            model_10d = self.LSTMModel()
            criterion = nn.MSELoss()
            optimizer_5d = torch.optim.Adam(model_5d.parameters())
            optimizer_10d = torch.optim.Adam(model_10d.parameters())

            for epoch in range(epochs):
                for seq, label_5d, label_10d in loader:
                    optimizer_5d.zero_grad()
                    pred_5d = model_5d(seq)
                    loss_5d = criterion(pred_5d, label_5d)
                    loss_5d.backward()
                    optimizer_5d.step()

                    optimizer_10d.zero_grad()
                    pred_10d = model_10d(seq)
                    loss_10d = criterion(pred_10d, label_10d)
                    loss_10d.backward()
                    optimizer_10d.step()

            self.models_5d[ticker] = model_5d
            self.models_10d[ticker] = model_10d
            self.scalers[ticker] = scaler

    def predict(self):
        predictions = []
        for ticker in self.tickers:
            data = _get_stock_data(ticker)[-self.window_size:]
            scaled = self.scalers[ticker].transform(data.values.reshape(-1, 1))

            with torch.no_grad():
                input_tensor = torch.FloatTensor(scaled).view(1, self.window_size, 1)
                pred_5d = self.models_5d[ticker](input_tensor)
                pred_10d = self.models_10d[ticker](input_tensor)

                predictions.append({
                    'ticker': ticker,
                    '5_day_return': self.scalers[ticker].inverse_transform(pred_5d.numpy())[0][0],
                    '10_day_return': self.scalers[ticker].inverse_transform(pred_10d.numpy())[0][0]
                })
        return pd.DataFrame(predictions)


if __name__ == "__main__":
    predictor = LSTMPredictor([])
    predictor.train('../../results/metrics/stock_tickers.csv')
    results = predictor.predict()
    save_results(results, '../../results/metrics/lstm_predictions.csv')
