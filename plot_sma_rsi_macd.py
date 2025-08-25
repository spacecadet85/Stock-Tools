import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD

def fetch_stock_data(ticker, period="6mo", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty or df['Close'].isna().all():
        raise ValueError(f"No data for ticker {ticker}. Is it delisted or incorrect?")
    return df

def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    # RSI
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # MACD
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

    # 1. SMA Crossover: 20-day crosses above 50-day
    if yesterday['SMA20'] < yesterday['SMA50'] and today['SMA20'] > today['SMA50']:
        signals['sma_crossover'] = True

    # 2. RSI: crosses above 30 (oversold recovery)
    if yesterday['RSI'] < 30 and today['RSI'] >= 30:
        signals['rsi_oversold'] = True

    # 3. MACD: crosses above MACD Signal line
    if yesterday['MACD'] < yesterday['MACD_SIGNAL'] and today['MACD'] > today['MACD_SIGNAL']:
        signals['macd_crossover'] = True

    # Require at least 2 of 3 signals to trigger a buy
    match_count = sum(signals.values())
    overall = match_count >= 2

    return overall, df.index[-1], signals

def plot_stock(df, ticker, buy_date=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1]})

    # --- Price + SMAs ---
    ax1.plot(df.index, df['Close'], label="Close Price", linewidth=1.5)
    ax1.plot(df.index, df['SMA20'], label="SMA20", linestyle='--')
    ax1.plot(df.index, df['SMA50'], label="SMA50", linestyle='--')
    if buy_date:
        ax1.axvline(x=buy_date, color='green', linestyle=':', linewidth=2, label='Buy Signal')
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} - Price, SMA, RSI, MACD")
    ax1.legend()
    ax1.grid(True)

    # --- RSI ---
    ax2.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
    ax2.axhline(y=30, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.grid(True)

    # --- MACD ---
    ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax3.plot(df.index, df['MACD_SIGNAL'], label='MACD Signal', color='orange')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_ylabel("MACD")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()

    # Save to file
    filename = f"{ticker}_sma_plot.png"
    plt.savefig(filename)
    print(f"📁 Plot with RSI/MACD saved to {filename}")

    # Show plot (optional)
    try:
        plt.show()
    except Exception:
        print("⚠️ Plot display failed (likely due to no GUI support)")

def print_signal_breakdown(signals):
    print("\n📊 Buy Signal Breakdown:")
    print(f"→ SMA20 crossover:     {'✔' if signals['sma_crossover'] else '✘'}")
    print(f"→ RSI recovery (>30):  {'✔' if signals['rsi_oversold'] else '✘'}")
    print(f"→ MACD crossover:      {'✔' if signals['macd_crossover'] else '✘'}")
    match_count = sum(signals.values())
    print(f"→ Signal Confidence:   {int((match_count / 3) * 100)}%\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 improved_sma.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    try:
        df = fetch_stock_data(ticker)
    except ValueError as e:
        print(f"❌ {e}")
        return

    df = calculate_indicators(df)
    signal, buy_date, signals = check_buy_signal(df)

    if signal:
        print(f"📈 Buy signal detected for {ticker} on {buy_date.date()}.")
        print_signal_breakdown(signals)
    else:
        print(f"No buy signal for {ticker} at this time.")

    plot_stock(df, ticker, buy_date if signal else None)

if __name__ == "__main__":
    main()
