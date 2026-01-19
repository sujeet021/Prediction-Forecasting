import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from math import sqrt
import pickle

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv('Bengaluru_House_Data.csv')
print('\nData shape:', df.shape)
print(df.head())

# -------------------------
# Basic EDA & Cleaning
# -------------------------
# 1. Quick info
print('\nInfo:')
df.info()

# 2. Drop duplicates
df.drop_duplicates(inplace=True)

# 3. Missing values summary
print('\nMissing values:')
print(df.isnull().sum())

# 4. Typical cleaning steps for this dataset (common in Bengaluru house price datasets):
# - Convert 'size' to numeric (e.g., '2 BHK' -> 2)
# - Convert 'total_sqft' to numeric (some entries like '2100 - 2850' -> take mean)
# - Convert 'price' (assumed in lakhs) to numeric; if provided per_sqft convert accordingly

# Example conversions (adjust according to actual column names in CSV)
if 'size' in df.columns:
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split()[0]) if pd.notnull(x) else np.nan)

# total_sqft cleaning
def sqft_to_num(x):
    try:
        if isinstance(x, str):
            if '-' in x:
                tokens = x.split('-')
                return (float(tokens[0].strip()) + float(tokens[1].strip()))/2
            if x.replace('.','',1).isdigit():
                return float(x)
            # handle sqft like '34.46Sq. Meter' -> try to extract numeric
            num = ''.join(ch for ch in x if (ch.isdigit() or ch=='.' or ch=='-'))
            if num:
                return float(num)
        if np.isnan(x):
            return np.nan
        return float(x)
    except:
        return np.nan

if 'total_sqft' in df.columns:
    df['total_sqft_num'] = df['total_sqft'].apply(sqft_to_num)

# price to numeric (assuming price column name is 'price')
if 'price' in df.columns:
    # if price is in lakhs, convert to numeric and to absolute (lakhs -> rupees*1e5) if desired
    df['price_num'] = pd.to_numeric(df['price'], errors='coerce')

# Drop rows with essential missing values
essential_cols = []
if 'price_num' in df.columns: essential_cols.append('price_num')
if 'total_sqft_num' in df.columns: essential_cols.append('total_sqft_num')
if 'bhk' in df.columns: essential_cols.append('bhk')

if len(essential_cols)>0:
    df.dropna(subset=essential_cols, inplace=True)

# Feature: price per sqft
if 'price_num' in df.columns and 'total_sqft_num' in df.columns:
    df['price_per_sqft'] = (df['price_num']*100000) / df['total_sqft_num']  # if price_num in lakhs

print('\nAfter cleaning shape:', df.shape)

# -------------------------
# Visualization (choose any 3 as requested)
# 1) Correlation heatmap
# 2) Boxplot for price_per_sqft by bhk
# 3) Pairplot (subset of numeric features) / swarmplot
# -------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt='.2f')
plt.title('Correlation heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# Boxplot: price_per_sqft by bhk (top 10 bhk categories or 1-6)
plt.figure(figsize=(10,6))
sns.boxplot(x='bhk', y='price_per_sqft', data=df[df['bhk']<=6])
plt.title('Boxplot: Price per sqft by BHK (<=6)')
plt.savefig('boxplot_bhk_pricepsqft.png')
plt.show()

# Pairplot: sample numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
pair_cols = [c for c in ['price_num','total_sqft_num','price_per_sqft','bhk'] if c in num_cols]
if len(pair_cols)>=2:
    sns.pairplot(df[pair_cols].sample(min(500, len(df))), diag_kind='kde')
    plt.savefig('pairplot.png')
    plt.show()

# Optional swarm plot (final swarm plot as requested) - for clarity, sample the data
if 'price_per_sqft' in df.columns:
    sample = df[df['bhk'].isin([1,2,3])].sample(min(300, len(df)))
    plt.figure(figsize=(10,6))
    sns.swarmplot(x='bhk', y='price_per_sqft', data=sample)
    plt.title('Swarm plot: price per sqft for BHK 1-3 (sample)')
    plt.savefig('swarmplot.png')
    plt.show()

# -------------------------
# Prepare data for machine learning (Regression)
# -------------------------
# Select features - keep numeric features: total_sqft_num, bhk, maybe 'bath' if present.
features = []
if 'total_sqft_num' in df.columns: features.append('total_sqft_num')
if 'bhk' in df.columns: features.append('bhk')
if 'bath' in df.columns: features.append('bath')

X = df[features]
y = df['price_num']

# Drop any rows where features are NaN
df = df.dropna(subset=features + ['price_num'])

# 🔹 Handle missing values properly before splitting data
from sklearn.impute import SimpleImputer

# Check again if any NaNs remain
print("Missing values before imputation:", X.isnull().sum().sum())

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert back to DataFrame
X = pd.DataFrame(X_imputed, columns=X.columns)

print("Missing values after imputation:", X.isnull().sum().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Machine Learning Models
# Train: LinearRegression, SVR, DecisionTreeRegressor, RandomForestRegressor, KNeighborsRegressor
# For Naive Bayes: we convert target to categories (bins) and run GaussianNB as classifier
# -------------------------
models = {
    'LinearRegression': LinearRegression(),
    'SVR': SVR(kernel='rbf'),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    #rmse = mean_squared_error(y_test, preds, squared=False)
    rmse= sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

# Naive Bayes as classification: bin price into categories (low, medium, high)
bins = [0, df['price_num'].quantile(0.33), df['price_num'].quantile(0.66), df['price_num'].max()+1]
labels = [0,1,2]
df['price_cat'] = pd.cut(df['price_num'], bins=bins, labels=labels)

# Use same features X (unscaled for NB is fine but I'll scale)
X_nb = scaler.fit_transform(df[features])
y_nb = df['price_cat'].astype(int)
Xnb_train, Xnb_test, ynb_train, ynb_test = train_test_split(X_nb, y_nb, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(Xnb_train, ynb_train)
ynb_pred = nb.predict(Xnb_test)
acc_nb = accuracy_score(ynb_test, ynb_pred)
print('\nGaussianNB (on binned price) accuracy:', acc_nb)
results['GaussianNB (binned)'] = {'accuracy': acc_nb}

# -------------------------
# Post-processing visualization: Compare any 3 models (we'll compare LinearRegression, RandomForest, KNN)
# -------------------------
compare_models = ['LinearRegression','RandomForest','KNN']
rmse_vals = [results[m]['rmse'] for m in compare_models]
mae_vals = [results[m]['mae'] for m in compare_models]
r2_vals = [results[m]['r2'] for m in compare_models]

plt.figure(figsize=(8,5))
plt.bar(compare_models, rmse_vals)
plt.title('RMSE comparison')
plt.ylabel('RMSE')
plt.savefig('rmse_comparison.png')
plt.show()

plt.figure(figsize=(8,5))
plt.bar(compare_models, mae_vals)
plt.title('MAE comparison')
plt.ylabel('MAE')
plt.savefig('mae_comparison.png')
plt.show()

plt.figure(figsize=(8,5))
plt.bar(compare_models, r2_vals)
plt.title('R2 comparison')
plt.ylabel('R2')
plt.savefig('r2_comparison.png')
plt.show()

# -------------------------
# Save best model (example: RandomForest)
# -------------------------
best_model = models['RandomForest']
with open('random_forest_model.pkl','wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler, 'features': features}, f)

print('\nSaved RandomForest model to random_forest_model.pkl')

# -------------------------
# Summary of results
# -------------------------
print('\nSummary of model results:')
for k,v in results.items():
    print(k, v)

# End of code
