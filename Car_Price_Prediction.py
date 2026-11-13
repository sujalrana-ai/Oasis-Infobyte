# %% [markdown]
# # Task 3: Car Price Prediction with Machine Learning
# **Objective:** Build a regression model that predicts the selling price of a used car based on features such as brand, car age, mileage, fuel type, and transmission.

# %%
# ==========================================
# 1. IMPORTS & DATASET LOADING
# ==========================================
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)

# Auto-detect whichever Cardekho CSV file is in the current folder
possible_files = [
    "car data.csv",
    "CAR DETAILS FROM CAR DEKHO.csv",
    "Car details v3.csv"
]

file_path = next((f for f in possible_files if os.path.exists(f)), None)
if file_path is None:
    raise FileNotFoundError(
        "Dataset not found! Place 'car data.csv', 'CAR DETAILS FROM CAR DEKHO.csv', "
        "or 'Car details v3.csv' in the same folder as this file."
    )

df = pd.read_csv(file_path)
print(f"Loaded file: {file_path}")
print("Initial Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

# %%
# ==========================================
# 2. DATA CLEANING & STANDARDIZATION
# ==========================================
# Standardize column names across Cardekho dataset variants
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

rename_map = {
    'name': 'car_name',
    'kms_driven': 'km_driven',
    'fuel': 'fuel_type'
}
df.rename(columns=rename_map, inplace=True)

# Keep core columns present across Cardekho CSV versions
core_features = [
    'car_name', 'year', 'selling_price', 'present_price',
    'km_driven', 'fuel_type', 'seller_type', 'transmission', 'owner'
]
df = df[[c for c in core_features if c in df.columns]].copy()

print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove nulls and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Fix inconsistent categorical casing (e.g., "petrol" vs "Petrol")
for col in ['fuel_type', 'seller_type', 'transmission']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Normalize selling_price to Lakhs INR if stored in raw Rupees
if df['selling_price'].median() > 1000:
    df['selling_price'] = df['selling_price'] / 100000.0

print("\nCleaned Shape:", df.shape)

# %%
# ==========================================
# 3. FEATURE ENGINEERING (CAR AGE & BRAND)
# ==========================================
current_year = datetime.datetime.now().year
df['car_age'] = current_year - df['year']

# Extract brand name from the first word of car_name
df['brand'] = df['car_name'].astype(str).str.strip().str.split().str[0].str.title()

# Group rare brands (< 5 listings) into 'Other' to prevent overfitting
brand_counts = df['brand'].value_counts()
rare_brands = brand_counts[brand_counts < 5].index
df['brand'] = df['brand'].replace(rare_brands, 'Other')

# Drop raw year and car_name columns
df_model = df.drop(columns=['year', 'car_name'])

print("\nTop Car Brands:\n", df_model['brand'].value_counts().head(10))
print("\nDescriptive Statistics:\n", df_model.describe().round(2))

# %% [markdown]
# ### Observation 1: Cleaning & Feature Engineering
# - Removed null rows and duplicate entries, and standardized text casing across categorical columns (`fuel_type`, `seller_type`, `transmission`).
# - Engineered `car_age` (`current_year - year`) and extracted `brand` from `car_name`, grouping rare brands under `Other`.

# %%
# ==========================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Distribution of Selling Prices
sns.histplot(df_model['selling_price'], bins=25, kde=True, color='teal', ax=axes[0])
axes[0].set_title("Distribution of Selling Price (Lakhs INR)", fontweight='bold')
axes[0].set_xlabel("Selling Price (Lakhs)")

# Plot 2: Selling Price vs. Fuel Type (Box Plot)
sns.boxplot(
    data=df_model, x='fuel_type', y='selling_price',
    hue='fuel_type', palette='Set2', legend=False, ax=axes[1]
)
axes[1].set_title("Selling Price vs. Fuel Type", fontweight='bold')
axes[1].set_xlabel("Fuel Type")
axes[1].set_ylabel("Selling Price (Lakhs)")

# Plot 3: Selling Price vs. Car Age (Scatter Plot)
sns.scatterplot(
    data=df_model, x='car_age', y='selling_price',
    hue='transmission', alpha=0.75, ax=axes[2]
)
axes[2].set_title("Selling Price vs. Car Age", fontweight='bold')
axes[2].set_xlabel("Car Age (Years)")
axes[2].set_ylabel("Selling Price (Lakhs)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation 2: Visual EDA Insights
# - **Price Distribution:** Right-skewed, with most used vehicles priced under ₹10 Lakhs.
# - **Fuel Type:** Diesel vehicles hold a noticeably higher median resale price than Petrol or CNG cars.
# - **Car Age & Transmission:** Selling price depreciates steadily as `car_age` increases, while Automatic cars command higher resale values than Manual cars.

# %%
# ==========================================
# 5. CATEGORICAL ENCODING & CORRELATION HEATMAP
# ==========================================
# One-Hot Encode categorical columns
df_encoded = pd.get_dummies(df_model, drop_first=True, dtype=int)

# Correlation heatmap for core features
core_cols = [c for c in df_encoded.columns if not c.startswith('brand_')]
plt.figure(figsize=(10, 7))
sns.heatmap(
    df_encoded[core_cols].corr(), annot=True, cmap='coolwarm',
    fmt=".2f", linewidths=0.5, vmin=-1, vmax=1
)
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# ==========================================
# 6. TRAIN/TEST SPLIT & REGRESSION MODELS
# ==========================================
X = df_encoded.drop(columns=['selling_price'])
y = df_encoded['selling_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2_Score": round(r2, 4)
    })
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values(by="R2_Score", ascending=False)
print("\nModel Evaluation Comparison:\n", results_df.to_string(index=False))

# %%
# ==========================================
# 7. FEATURE IMPORTANCE CHART (BEST MODEL)
# ==========================================
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\nBest Performing Model: {best_model_name}")

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
else:
    importances = pd.Series(np.abs(best_model.coef_), index=X.columns)

top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=top_features.values, y=top_features.index,
    hue=top_features.index, palette='viridis', legend=False
)
plt.title(f"Top 10 Most Important Features ({best_model_name})", fontsize=13, fontweight='bold')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Final Conclusion
# - **Best Model:** Tree-based ensemble models (**Random Forest** / **Gradient Boosting**) achieve lower MAE/RMSE and higher $R^2$ scores than Linear Regression by capturing non-linear depreciation and brand-tier interactions.
# - **Top Predictors:** Showroom/present price (if available in the dataset variant), `car_age`, `km_driven`, and `fuel_type` (Diesel) are the strongest drivers of used car selling price.
