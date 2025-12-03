# weather_visualizer.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

DATA_PATH = Path("data/weather.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

def generate_sample_data(path=DATA_PATH, days=365):
    rng = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='D')
    df = pd.DataFrame({
        "date": rng,
        "temperature_c": 20 + 8*np.sin(np.linspace(0, 3.14*2, days)) + np.random.normal(0,2,days),
        "rain_mm": np.random.gamma(1.2, 2.0, size=days),
        "humidity_pct": np.clip(50 + 20*np.sin(np.linspace(1,4,days)) + np.random.normal(0,6,days), 0, 100)
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sample data written to {path}")
    return path

def load_and_inspect(path=DATA_PATH):
    if not path.exists():
        print("No data found. Creating sample dataset.")
        generate_sample_data(path)
    df = pd.read_csv(path)
    print("HEAD:")
    print(df.head())
    print("\nINFO:")
    print(df.info())
    return df

def clean_data(df: pd.DataFrame):
    # Ensure date column
    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        raise KeyError("Expected 'date' column in dataset.")
    # Fill or drop missing
    df = df.dropna(subset=['date'])
    # numeric conversions
    for col in ['temperature_c', 'rain_mm', 'humidity_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill small NaNs with interpolation
    df[['temperature_c','rain_mm','humidity_pct']] = df[['temperature_c','rain_mm','humidity_pct']].interpolate().fillna(method='bfill').fillna(method='ffill')
    df = df.sort_values('date').set_index('date')
    return df

def compute_stats(df):
    # daily already
    monthly = df.resample('M').agg({
        'temperature_c': ['mean','min','max','std'],
        'rain_mm': 'sum',
        'humidity_pct': ['mean','min','max']
    })
    yearly = df.resample('Y').agg({
        'temperature_c': 'mean',
        'rain_mm': 'sum',
        'humidity_pct': 'mean'
    })
    return monthly, yearly

def plot_temperature_trend(df):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df['temperature_c'])
    plt.title("Daily Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    out = OUT_DIR / "temp_trend.png"
    plt.savefig(out)
    plt.close()
    print("Saved", out)

def plot_monthly_rainfall(df):
    monthly_rain = df['rain_mm'].resample('M').sum()
    plt.figure(figsize=(10,4))
    monthly_rain.plot(kind='bar')
    plt.title("Monthly Rainfall Totals")
    plt.xlabel("Month")
    plt.ylabel("Rainfall (mm)")
    plt.tight_layout()
    out = OUT_DIR / "rainfall_monthly.png"
    plt.savefig(out)
    plt.close()
    print("Saved", out)

def plot_humidity_vs_temp(df):
    plt.figure(figsize=(6,6))
    plt.scatter(df['temperature_c'], df['humidity_pct'], alpha=0.6)
    plt.title("Humidity vs Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.tight_layout()
    out = OUT_DIR / "humidity_vs_temp.png"
    plt.savefig(out)
    plt.close()
    print("Saved", out)

def export_cleaned(df):
    out = OUT_DIR / "cleaned_weather.csv"
    df.reset_index().to_csv(out, index=False)
    print("Exported cleaned data to", out)

def main():
    df_raw = load_and_inspect()
    df_clean = clean_data(df_raw)
    monthly, yearly = compute_stats(df_clean)
    print("\nMonthly summary (head):")
    print(monthly.head())
    plot_temperature_trend(df_clean)
    plot_monthly_rainfall(df_clean)
    plot_humidity_vs_temp(df_clean)
    export_cleaned(df_clean)
    # save summary markdown
    summary = OUT_DIR / "summary.txt"
    with summary.open("w") as f:
        f.write("Weather Data Summary\n")
        f.write("====================\n\n")
        f.write(f"Total days analyzed: {len(df_clean)}\n")
        f.write("Yearly summary:\n")
        f.write(str(yearly))
    print("Summary written to", summary)

if __name__ == "__main__":
    main()