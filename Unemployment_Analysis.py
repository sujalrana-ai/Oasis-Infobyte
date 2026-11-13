# %% [markdown]
# # Task 2: Unemployment Analysis in India (COVID-19 Impact)
# **Objective:** Perform exploratory data analysis on unemployment data to uncover regional and temporal trends, focusing on the impact of the COVID-19 pandemic in India.

# %%
# ==========================================
# 1. IMPORTS & DATA LOADING
# ==========================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Auto-detect whichever Kaggle CSV file is in the current folder
possible_files = [
    "Unemployment_Rate_upto_11_2020.csv",
    "Unemployment in India.csv"
]

file_path = next((f for f in possible_files if os.path.exists(f)), None)
if file_path is None:
    raise FileNotFoundError(
        "Dataset not found! Place 'Unemployment_Rate_upto_11_2020.csv' or "
        "'Unemployment in India.csv' in the same folder as this file."
    )

df = pd.read_csv(file_path)
print(f"Loaded file: {file_path}")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

# %%
# ==========================================
# 2. DATA CLEANING, NULL CHECK & TYPE CONVERSION
# ==========================================
# Strip leading/trailing whitespace from column headers
df.columns = df.columns.str.strip()

# Standardize column names across both Kaggle CSV versions
rename_map = {
    'Region': 'State',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate',
    'Region.1': 'Zone',
    'Area': 'Area_Type'
}
df.rename(columns=rename_map, inplace=True)

print("\nMissing Values Before Cleaning:\n", df.isnull().sum())
print("\nData Types Before Conversion:\n", df.dtypes)

# Drop missing rows and reset index
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Clean State names and convert Date to datetime
df['State'] = df['State'].astype(str).str.strip()
df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), dayfirst=True)

# Extract temporal features
df['Year'] = df['Date'].dt.year
df['Month_Num'] = df['Date'].dt.month
df['Month'] = df['Date'].dt.strftime('%b %Y')

print("\nCleaned Dataset Shape:", df.shape)
print("\nData Types After Conversion:\n", df.dtypes)
print("\nDescriptive Statistics:\n", df[['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']].describe().round(2))

# %%
# ==========================================
# 3. EDA: REGION-WISE & MONTH-WISE TRENDS
# ==========================================
group_col = 'Zone' if 'Zone' in df.columns else 'Area_Type'
region_avg = df.groupby(group_col)['Unemployment_Rate'].mean().sort_values(ascending=False)
print(f"\nAverage Unemployment Rate by {group_col} (%):\n", region_avg.round(2))

month_avg = (
    df.groupby(['Year', 'Month_Num', 'Month'])['Unemployment_Rate']
    .mean()
    .reset_index()
    .sort_values(['Year', 'Month_Num'])
)

plt.figure(figsize=(12, 5))
sns.lineplot(
    data=month_avg,
    x='Month',
    y='Unemployment_Rate',
    marker='o',
    linewidth=2.5,
    color='crimson'
)
plt.title("All-India Monthly Average Unemployment Rate", fontsize=14, fontweight='bold')
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 4. TIME-SERIES LINE CHART (MAJOR STATES)
# ==========================================
selected_states = ['Maharashtra', 'Gujarat', 'Uttar Pradesh', 'Tamil Nadu']
state_df = (
    df[df['State'].isin(selected_states)]
    .groupby(['Date', 'State'])['Unemployment_Rate']
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=state_df,
    x='Date',
    y='Unemployment_Rate',
    hue='State',
    marker='o',
    linewidth=2
)
plt.title("Unemployment Rate Over Time Across Major Indian States", fontsize=14, fontweight='bold')
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Date")
plt.legend(title="State")
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 5. BAR CHART: TOP 10 STATES BY UNEMPLOYMENT
# ==========================================
top_10_states = (
    df.groupby('State')['Unemployment_Rate']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(11, 6))
sns.barplot(
    data=top_10_states,
    x='Unemployment_Rate',
    y='State',
    hue='State',
    palette='viridis',
    legend=False
)
plt.title("Top 10 Indian States by Average Unemployment Rate", fontsize=14, fontweight='bold')
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 6. CORRELATION HEATMAP
# ==========================================
corr_cols = ['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, vmin=-1, vmax=1)
plt.title("Correlation Heatmap: Unemployment, Employment & Labour Participation", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 7. PRE-COVID VS. POST-COVID COMPARISON
# ==========================================
df['Period'] = np.where(
    df['Date'] < '2020-03-24',
    'Pre-COVID (Before Mar 24, 2020)',
    'Post-COVID (After Mar 24, 2020)'
)

covid_comparison = (
    df.groupby('Period')[['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']]
    .mean()
    .reset_index()
)
print("\nPre-COVID vs. Post-COVID Summary:\n", covid_comparison.round(2))

plt.figure(figsize=(8, 5))
sns.barplot(
    data=covid_comparison,
    x='Period',
    y='Unemployment_Rate',
    hue='Period',
    palette=['#2ecc71', '#e74c3c'],
    legend=False
)
plt.title("Pre-COVID vs. Post-COVID Average Unemployment Rate in India", fontsize=13, fontweight='bold')
plt.ylabel("Mean Unemployment Rate (%)")
plt.xlabel("")
plt.tight_layout()
plt.show()
