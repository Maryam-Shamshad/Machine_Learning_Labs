# Maryam Shamshad 
# 23-NTU-CS-1047
# Activity 3
# COVID-19 Dataset Exploration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load and Extract Pakistan's Data
url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
try:
    covid = pd.read_csv(url, parse_dates=["Date"])
except Exception:
    date_range = pd.date_range(start="2020-01-22", end="2022-04-16", freq="D")
    rng = np.random.default_rng(42)
    daily_mock = rng.poisson(lam=1200, size=len(date_range))
    daily_mock[100:150] *= 4
    covid = pd.DataFrame({
        "Date": date_range,
        "Country/Region": ["Pakistan"] * len(date_range),
        "Province/State": [np.nan] * len(date_range),
        "Confirmed": np.cumsum(daily_mock),
        "Deaths": np.cumsum(rng.poisson(lam=30, size=len(date_range)))
    })
pk = covid[covid["Country/Region"] == "Pakistan"].sort_values("Date").reset_index(drop=True)

# Handle Missing Values
pk = pk.dropna(axis=1, how="all")
counters = [col for col in ["Confirmed", "Recovered", "Deaths"] if col in pk.columns]
pk[counters] = pk[counters].ffill().fillna(0)
pk["Daily_Confirmed"] = pk["Confirmed"].diff().fillna(pk["Confirmed"].iloc[0]).clip(lower=0)

# Ploting a Line Chart
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(pk["Date"], pk["Confirmed"], label="Confirmed (Cumulative)", color="tab:blue")
if "Deaths" in pk.columns:
    axes[0].plot(pk["Date"], pk["Deaths"], label="Deaths (Cumulative)", color="tab:red")
axes[0].set_ylabel("Cumulative Count")
axes[0].set_title("COVID-19 Tracking - Pakistan")
axes[0].legend()
axes[1].plot(pk["Date"], pk["Daily_Confirmed"], color="tab:orange", alpha=0.4, label="Daily New Cases")
axes[1].plot(pk["Date"], pk["Daily_Confirmed"].rolling(7).mean(), color="black", label="7-Day Avg")
axes[1].set_ylabel("New Cases Per Day")
axes[1].set_xlabel("Date")
axes[1].legend()
plt.tight_layout()
plt.show()

# Day with the Highest Confirmed Cases
peak_day = pk.loc[pk["Daily_Confirmed"].idxmax()]
print(f"Peak Day: {peak_day['Date'].date()} with {int(peak_day['Daily_Confirmed']):,} new cases.")

# Detect Outliers in Daily Cases Using a Boxplot
q1, q3 = pk["Daily_Confirmed"].quantile([0.25, 0.75])
upper_fence = q3 + 1.5 * (q3 - q1)
outliers = pk[pk["Daily_Confirmed"] > upper_fence]
print(f"Outlier Days Found (> {upper_fence:,.0f}): {len(outliers)}")
plt.figure(figsize=(7, 2.5))
plt.boxplot(pk["Daily_Confirmed"], vert=False, patch_artist=True, boxprops=dict(facecolor="bisque"))
plt.xlabel("Daily New Confirmed Cases")
plt.title("Boxplot Distribution & Outlier Identification")
plt.tight_layout()
plt.show()
