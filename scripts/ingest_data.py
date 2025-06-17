import pandas as pd

def load_and_transform(file_path):
    """
    Loads raw BTC data from a CSV file and ensures consistent formatting.
    Assumes the CSV contains 'timestamp' and 'price' columns.
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'])

    # Enforce expected schema
    df = df.rename(columns={"Datetime": "timestamp", "Close": "price"}) if 'Datetime' in df.columns else df
    df = df[['timestamp', 'price']] if all(col in df.columns for col in ['timestamp', 'price']) else df

    df = df.sort_values('timestamp')
    df = df.reset_index(drop=True)

    return df
