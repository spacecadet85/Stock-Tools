import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
import yaml
from pathlib import Path

def fetch_stock_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty or df['Close'].isna().all():
        raise ValueError(f"No data for ticker {ticker}. Is it delisted or incorrect?")
    return df

def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()

    return df

def check_buy_signal(df):
    if len(df) < 2:
        return False, None, {}

    yesterday = df.iloc[-2]
    today = df.iloc[-1]

    signals = {
        'sma_crossover': False,
        'rsi_oversold': False,
        'macd_crossover': False,
    }

    if yesterday['SMA20'] < yesterday['SMA50'] and today['SMA20'] > today['SMA50']:
        signals['sma_crossover'] = True

    if yesterday['RSI'] < 30 and today['RSI'] >= 30:
        signals['rsi_oversold'] = True

    if yesterday['MACD'] < yesterday['MACD_SIGNAL'] and today['MACD'] > today['MACD_SIGNAL']:
        signals['macd_crossover'] = True

    match_count = sum(signals.values())
    overall = match_count >= 2

    return overall, df.index[-1], signals

def plot_stock(df, ticker, buy_date=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1]})

    ax1.plot(df.index, df['Close'], label="Close Price", linewidth=1.5)
    ax1.plot(df.index, df['SMA20'], label="SMA20", linestyle='--')
    ax1.plot(df.index, df['SMA50'], label="SMA50", linestyle='--')
    if buy_date:
        ax1.axvline(x=buy_date, color='green', linestyle=':', linewidth=2, label='Buy Signal')
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} - Price, SMA, RSI, MACD")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
    ax2.axhline(y=30, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax3.plot(df.index, df['MACD_SIGNAL'], label='MACD Signal', color='orange')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_ylabel("MACD")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    filename = f"{ticker}_sma_plot.png"
    plt.savefig(filename)
    print(f"📁 Plot saved to {filename}")
    plt.close()

def print_signal_breakdown(ticker, buy_date, signals):
    print(f"\n📈 Buy signal detected for {ticker} on {buy_date.date()}.\n")
    print(f"📊 Buy Signal Breakdown:")
    print(f"→ SMA20 crossover:     {'✔' if signals['sma_crossover'] else '✘'}")
    print(f"→ RSI recovery (>30):  {'✔' if signals['rsi_oversold'] else '✘'}")
    print(f"→ MACD crossover:      {'✔' if signals['macd_crossover'] else '✘'}")
    match_count = sum(signals.values())
    print(f"→ Signal Confidence:   {int((match_count / 3) * 100)}%\n")

def load_tickers_from_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get("tickers", [])

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 improved_sma.py tickers.yaml")
        sys.exit(1)

    yaml_file = sys.argv[1]
    if not Path(yaml_file).exists():
        print(f"YAML file not found: {yaml_file}")
        sys.exit(1)

    tickers = load_tickers_from_yaml(yaml_file)
    if not tickers:
        print("No tickers found in YAML file.")
        sys.exit(1)

    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n📡 Analyzing {ticker}...")
        try:
            df = fetch_stock_data(ticker)
            df = calculate_indicators(df)
            signal, buy_date, signals = check_buy_signal(df)
            if signal:
                print_signal_breakdown(ticker, buy_date, signals)
            else:
                print(f"❌ No buy signal for {ticker} at this time.")
            plot_stock(df, ticker, buy_date if signal else None)
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()
