import pandas as pd

def engineer_features(df):
    """
    Applies feature transformations on the BTC price data.
    Assumes input DataFrame contains columns: ['timestamp', 'price']
    """
    df = df.copy()
    df = df.sort_values('timestamp')

    # Feature engineering
    df['return_1h'] = df['price'].pct_change()
    df['rolling_mean_3h'] = df['price'].rolling(window=3, min_periods=1).mean()
    df['rolling_mean_6h'] = df['price'].rolling(window=6, min_periods=1).mean()
    df['rolling_std_3h'] = df['price'].rolling(window=3, min_periods=1).std()

    # Add future price for classification targets (optional downstream)
    df['future_price'] = df['price'].shift(-1)

    # Clean up
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
