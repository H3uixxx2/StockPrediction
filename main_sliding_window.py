import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from mamba import Mamba, MambaConfig
import matplotlib.pyplot as plt

# --- Cấu hình phiên bản GỐC (Trước STB) ---
class Args:
    use_cuda = False
    seed = 42
    epochs = 50
    lr = 0.0005         
    hidden = 64         
    layer = 2
    window_size = 60    
    prediction_horizon = 5 
    batch_size = 64
    ts_code = '600036'
    initial_capital = 10000.0
    transaction_fee = 0.001
    test_days = 300
    threshold = 0.005   

args = Args()
args.cuda = args.use_cuda and torch.cuda.is_available()

if args.cuda:
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print(">>> Đang sử dụng GPU Apple Silicon (MPS)!")
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(args.seed)

# --- Indicators ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

# --- Data Preparation ---
def prepare_data():
    print(f">>> Đang đọc dữ liệu mã {args.ts_code}...")
    df = pd.read_csv(f'{args.ts_code}.SH.csv')
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    close = df['close']
    df['log_ret'] = np.log(close / close.shift(1))
    df['SMA_5'] = close.rolling(window=5).mean() / close
    df['SMA_20'] = close.rolling(window=20).mean() / close
    df['RSI'] = compute_rsi(close) / 100.0
    macd, signal = compute_macd(close)
    df['MACD'] = macd / close
    df['MACD_Signal'] = signal / close
    bb_upper, bb_lower = compute_bollinger_bands(close)
    df['BB_Upper'] = (bb_upper - close) / close
    df['BB_Lower'] = (close - bb_lower) / close
    df['Volat_20'] = close.rolling(window=20).std() / close
    df['Vol_Change'] = df['vol'].pct_change()
    
    df.dropna(inplace=True)
    
    feature_cols = ['log_ret', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 
                    'BB_Upper', 'BB_Lower', 'Volat_20', 'Vol_Change']
    
    features = df[feature_cols].values
    mean = np.mean(features, axis=0); std = np.std(features, axis=0)
    features_norm = (features - mean) / (std + 1e-8)
    
    future_close = close.shift(-args.prediction_horizon)
    returns_horizon = (future_close.loc[df.index] - close.loc[df.index]) / close.loc[df.index]
    labels = (returns_horizon > args.threshold).astype(int).values
    
    valid_mask = ~np.isnan(returns_horizon)
    features_norm = features_norm[valid_mask]
    labels = labels[valid_mask]
    dates = df['trade_date'][valid_mask].values
    close_prices = df['close'][valid_mask].values
    
    X, y = [], []
    for i in range(len(features_norm) - args.window_size):
        X.append(features_norm[i : i + args.window_size])
        y.append(labels[i + args.window_size])
    X = np.array(X); y = np.array(y)
    
    test_size = args.test_days
    train_val_size = len(X) - test_size
    val_size = int(train_val_size * 0.15)
    train_size = train_val_size - val_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    
    test_dates_out = dates[args.window_size:][-test_size:]
    test_close_out = close_prices[args.window_size:][-test_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (test_dates_out, test_close_out), features_norm.shape[1]

# --- Model ---
class MambaClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers):
        super().__init__()
        self.config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.mamba = Mamba(self.config)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba(x)
        return self.head(x[:, -1, :]).flatten()

# --- Train ---
def train_model(model, train_loader, val_loader):
    pos_weight = torch.tensor([1.5]).to(device) # Trọng số cao cho lớp Tăng
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0; best_state = None; patience = 20; counter = 0
    print(">>> Bắt đầu Training...")
    for epoch in range(args.epochs):
        model.train()
        correct = 0; total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch); loss = criterion(logits, y_batch)
            loss.backward(); optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_batch).sum().item(); total += y_batch.size(0)
        
        model.eval()
        v_correct = 0; v_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                v_correct += (preds == y_batch).sum().item(); v_total += y_batch.size(0)
        
        val_acc = v_correct / v_total
        scheduler.step(val_acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Acc: {correct/total:.2%} | Val Acc: {val_acc:.2%}")
        if val_acc > best_acc:
            best_acc = val_acc; best_state = model.state_dict(); counter = 0
        else:
            counter += 1
            if counter >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model

# --- Evaluate (Logic GỐC - Lãi to) ---
def evaluate(model, test_loader, dates, close_prices):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
    preds = (np.array(all_probs) > 0.5).astype(int)
    
    print("\n" + "="*40)
    print(f"KẾT QUẢ CLASSIFICATION (Test Set) - Mã {args.ts_code}")
    print("="*40)
    
    cash = args.initial_capital; shares = 0; position = 0; history = []
    entry_price = 0; stop_loss_pct = 0.02; take_profit_pct = 0.05
    
    for i in range(len(preds) - 1):
        price = close_prices[i]
        next_price = close_prices[i+1]
        signal = preds[i]
        
        if position == 1:
            current_ret = (price - entry_price) / entry_price
            if current_ret <= -stop_loss_pct or current_ret >= take_profit_pct:
                cash = shares * price * (1 - args.transaction_fee)
                shares = 0; position = 0; signal = 0 # Ép bán
        
        if signal == 1 and position == 0:
            cash = cash * (1 - args.transaction_fee)
            shares = cash / price
            entry_price = price; position = 1
        elif signal == 0 and position == 1:
            cash = shares * price * (1 - args.transaction_fee)
            shares = 0; position = 0
            
        history.append(cash + shares * next_price)
        
    final_nav = history[-1] if history else args.initial_capital
    bh_shares = (args.initial_capital * (1 - args.transaction_fee)) / close_prices[0]
    bh_final = (bh_shares * close_prices[-2]) * (1 - args.transaction_fee)
    
    print(f"Trading Profit: ${(final_nav - args.initial_capital):.2f} ({(final_nav/args.initial_capital - 1)*100:.2f}%)")
    print(f"Buy&Hold Profit: ${(bh_final - args.initial_capital):.2f} ({(bh_final/args.initial_capital - 1)*100:.2f}%)")

if __name__ == "__main__":
    pack = prepare_data()
    if pack:
        (xt, yt), (xv, yv), (xe, ye), (tdates, tc), idim = pack
        trl = DataLoader(TensorDataset(torch.from_numpy(xt).float(), torch.from_numpy(yt).float()), batch_size=args.batch_size, shuffle=True)
        vll = DataLoader(TensorDataset(torch.from_numpy(xv).float(), torch.from_numpy(yv).float()), batch_size=args.batch_size)
        tsl = DataLoader(TensorDataset(torch.from_numpy(xe).float(), torch.from_numpy(ye).float()), batch_size=args.batch_size, shuffle=False)
        m = train_model(MambaClassifier(idim, args.hidden, args.layer).to(device), trl, vll)
        evaluate(m, tsl, tdates, tc)
