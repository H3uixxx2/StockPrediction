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
    ts_code = '601988'
    initial_capital = 10000.0
    transaction_fee = 0.001
    test_days = 300 # Giữ lại biến này dù logic mới sẽ dùng date split
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

# --- Data Preparation (FIXED CRITICAL BUGS) ---
def prepare_data():
    print(f">>> Đang đọc dữ liệu mã {args.ts_code}...")
    df = pd.read_csv(f'{args.ts_code}.SH.csv')
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 1. Feature Engineering (Tính toán chỉ báo trên toàn bộ chuỗi để tránh bị ngắt quãng ở biên)
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
    
    # Labeling (Future Return) - Đây là Target, không dùng làm Feature
    future_close = close.shift(-args.prediction_horizon)
    returns_horizon = (future_close - close) / close
    df['label'] = (returns_horizon > args.threshold).astype(int)
    
    # Loại bỏ NaN do Rolling và Shift
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    feature_cols = ['log_ret', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 
                    'BB_Upper', 'BB_Lower', 'Volat_20', 'Vol_Change']
    
    # 2. Chronological Split (Chia theo thời gian thực)
    # Train: 2007 -> 2018
    # Val:   2019 -> 2020
    # Test:  2021 -> 2022 (End)
    
    train_mask = (df['trade_date'].dt.year >= 2007) & (df['trade_date'].dt.year <= 2018)
    val_mask   = (df['trade_date'].dt.year >= 2019) & (df['trade_date'].dt.year <= 2020)
    test_mask  = (df['trade_date'].dt.year >= 2021)
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df   = df[val_mask].reset_index(drop=True)
    test_df  = df[test_mask].reset_index(drop=True)
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    
    # 3. Normalization (QUAN TRỌNG: Chỉ fit trên Train, rồi transform Val/Test)
    # Tránh Data Leakage: Không được biết Mean/Std của tương lai
    X_train_raw = train_df[feature_cols].values
    
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    
    # Hàm normalize
    def normalize(data_raw, mean, std):
        return (data_raw - mean) / (std + 1e-8)
        
    X_train_norm = normalize(train_df[feature_cols].values, mean, std)
    X_val_norm   = normalize(val_df[feature_cols].values, mean, std)
    X_test_norm  = normalize(test_df[feature_cols].values, mean, std)
    
    # 4. Create Sliding Windows
    # FIX: Truyền trực tiếp cột date và close từ mỗi split, KHÔNG dùng index để tra cứu ngược
    def create_dataset(X_norm, labels, dates_col, close_col, window_size):
        Xs, ys, out_dates, out_prices = [], [], [], []
        n = len(X_norm)
        for i in range(n - window_size):
            Xs.append(X_norm[i : i + window_size])
            ys.append(labels[i + window_size])
            out_dates.append(dates_col[i + window_size])  # Trực tiếp lấy ngày từ cột
            out_prices.append(close_col[i + window_size]) # Trực tiếp lấy giá từ cột
            
        return np.array(Xs), np.array(ys), out_dates, np.array(out_prices)

    X_train, y_train, _, _ = create_dataset(
        X_train_norm, train_df['label'].values, 
        train_df['trade_date'].values, train_df['close'].values, 
        args.window_size
    )
    X_val, y_val, _, _ = create_dataset(
        X_val_norm, val_df['label'].values, 
        val_df['trade_date'].values, val_df['close'].values, 
        args.window_size
    )
    X_test, y_test, test_dates, test_prices = create_dataset(
        X_test_norm, test_df['label'].values, 
        test_df['trade_date'].values, test_df['close'].values, 
        args.window_size
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (test_dates, test_prices), len(feature_cols)

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
    pos_weight = torch.tensor([1.2]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0; best_state = None; patience = 15; counter = 0
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
        
        train_acc = correct/total if total > 0 else 0
        
        model.eval()
        v_correct = 0; v_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                v_correct += (preds == y_batch).sum().item(); v_total += y_batch.size(0)
        
        val_acc = v_correct / v_total if v_total > 0 else 0
        scheduler.step(val_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
            
        if val_acc > best_acc:
            best_acc = val_acc; best_state = model.state_dict(); counter = 0
        else:
            counter += 1
            if counter >= patience: 
                print(f"Early stop at epoch {epoch}")
                break
                
    if best_state: model.load_state_dict(best_state)
    return model

# --- Evaluate (FIXED - Realistic Backtest) ---
def evaluate(model, test_loader, dates, close_prices, y_true):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
    preds = (np.array(all_probs) > 0.5).astype(int)
    
    # Tính Accuracy thực sự
    accuracy = (preds == y_true).mean()
    
    print("\n" + "="*50)
    print(f"KẾT QUẢ CLASSIFICATION (Stress Test 2021-2022) - Mã {args.ts_code}")
    print("="*50)
    
    # Format dates safely
    try:
        start_date = pd.to_datetime(dates[0]).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')
    except:
        start_date = str(dates[0])
        end_date = str(dates[-1])
    
    print(f"Test Period: {start_date} -> {end_date}")
    print(f"Test Accuracy: {accuracy:.2%} (True prediction performance)")
    
    # Backtest Logic - REALISTIC
    # Tín hiệu T (Close) -> Mua tại giá Open T+1 (gần đúng = Close T+1 để đơn giản hóa)
    # Nghĩa là: signal[i] -> Thực hiện giao dịch với price[i+1]
    
    cash = args.initial_capital
    shares = 0
    position = 0  # 0 = không giữ, 1 = đang giữ
    history = [args.initial_capital]
    entry_price = 0
    
    for i in range(len(preds) - 1):
        signal = preds[i]
        
        # Giá thực hiện là giá ngày HÔM SAU (realistic execution)
        exec_price = close_prices[i + 1]
        
        # Stoploss/Takeprofit check (dựa trên giá hiện tại)
        if position == 1:
            current_ret = (close_prices[i] - entry_price) / entry_price
            if current_ret <= -0.05 or current_ret >= 0.10:
                cash = shares * exec_price * (1 - args.transaction_fee)
                shares = 0
                position = 0
                history.append(cash)
                continue  # Đã bán, bỏ qua signal
        
        if signal == 1 and position == 0:  # Buy signal
            cost = cash * args.transaction_fee
            cash = cash - cost
            shares = cash / exec_price
            entry_price = exec_price
            position = 1
        elif signal == 0 and position == 1:  # Sell signal
            cash = shares * exec_price * (1 - args.transaction_fee)
            shares = 0
            position = 0
            
        # NAV cuối ngày
        current_nav = cash + shares * close_prices[min(i + 1, len(close_prices) - 1)]
        history.append(current_nav)
        
    final_nav = history[-1]
    
    # Buy & Hold Benchmark
    bh_shares = (args.initial_capital * (1 - args.transaction_fee)) / close_prices[0]
    bh_final = bh_shares * close_prices[-1] * (1 - args.transaction_fee)
    
    trading_ret = (final_nav / args.initial_capital - 1) * 100
    bh_ret = (bh_final / args.initial_capital - 1) * 100
    
    print(f"Trading Profit: ${(final_nav - args.initial_capital):.2f} ({trading_ret:.2f}%)")
    print(f"Buy&Hold Profit: ${(bh_final - args.initial_capital):.2f} ({bh_ret:.2f}%)")
    print(f"Alpha (Trading - B&H): {trading_ret - bh_ret:.2f}%")

if __name__ == "__main__":
    pack = prepare_data()
    if pack:
        (xt, yt), (xv, yv), (xe, ye), (tdates, tc), idim = pack
        
        trl = DataLoader(TensorDataset(torch.from_numpy(xt).float(), torch.from_numpy(yt).float()), batch_size=args.batch_size, shuffle=True)
        vll = DataLoader(TensorDataset(torch.from_numpy(xv).float(), torch.from_numpy(yv).float()), batch_size=args.batch_size)
        tsl = DataLoader(TensorDataset(torch.from_numpy(xe).float(), torch.from_numpy(ye).float()), batch_size=args.batch_size, shuffle=False)
        
        m = train_model(MambaClassifier(idim, args.hidden, args.layer).to(device), trl, vll)
        evaluate(m, tsl, tdates, tc, ye)  # Truyền thêm y_true để tính accuracy
