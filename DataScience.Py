import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

print(">>> [1/6] LOADING ENTERPRISE DATA...")
df_customers = pd.read_csv('customers.csv')
df_products = pd.read_csv('products.csv')
df_sales = pd.read_csv('sales_transactions.csv')

df_sales = df_sales.drop_duplicates()
df_customers = df_customers.dropna(subset=['customer_id', 'age'])
df_sales = df_sales.dropna(subset=['order_id', 'quantity'])
df_products = df_products[df_products['unit_price'] > 0]
df_sales = df_sales[df_sales['quantity'] > 0]
print(f"[*] Cleaned data: {len(df_sales)} sales records remaining.")


print(">>> [2/6] EXECUTING DATA INTEGRATION & FEATURE ENGINEERING...")

df_master = df_sales.merge(df_customers, on='customer_id', how='left') \
                    .merge(df_products, on='product_id', how='left')

df_master['revenue'] = df_master['quantity'] * df_master['unit_price']
df_master['order_date'] = pd.to_datetime(df_master['order_date'])
df_master['month'] = df_master['order_date'].dt.month

print(f"[*] Pipeline Success: Processed {df_master.shape[0]} transaction records.")
print(df_master[['order_date', 'product_name', 'quantity', 'revenue']].head())

print("\n>>> [3/6] GENERATING VISUAL ANALYTICS...")
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='category', y='revenue', data=df_master, estimator=sum, errorbar=None, palette='mako')
plt.title('Total Revenue by Product Category (Samsung)', fontweight='bold')
plt.ylabel('Total Revenue (USD)')
plt.xlabel('Category')

plt.subplot(1, 2, 2)
sns.histplot(df_master['age'], bins=20, kde=True, color='#285291')
plt.title('Customer Age Distribution', fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print("\n>>> [4/6] RUNNING K-MEANS CLUSTERING...")

snapshot_date = df_master['order_date'].max() + timedelta(days=1)
rfm = df_master.groupby('customer_id').agg({
    'order_date': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'revenue': 'sum'
}).reset_index()
rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm[['recency', 'frequency', 'monetary']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_centers = rfm.groupby('cluster')['monetary'].mean().sort_values()
rfm['segment_label'] = rfm['cluster'].map({
    cluster_centers.index[2]: 'Tier 1: VIP Clients',
    cluster_centers.index[1]: 'Tier 2: Standard',
    cluster_centers.index[0]: 'Tier 3: Churn Risk'
})

print("[*] Top 5 Segmented Customers:")
print(rfm[['customer_id', 'recency', 'monetary', 'segment_label']].head())

print("\n>>> [5/6] TRAINING PREDICTIVE ML MODELS...")

df_ml = df_master.copy()

le_gender = LabelEncoder()
df_ml['gender'] = le_gender.fit_transform(df_ml['gender'])

le_category = LabelEncoder()
df_ml['category'] = le_category.fit_transform(df_ml['category'])

features = ['age', 'gender', 'category', 'month', 'quantity']
X = df_ml[features]
y = df_ml['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_ml = StandardScaler()
X_train_scaled = scaler_ml.fit_transform(X_train)
X_test_scaled = scaler_ml.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_r2 = r2_score(y_test, rf_predictions)

dl_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
dl_model.fit(X_train_scaled, y_train)
dl_predictions = dl_model.predict(X_test_scaled)
dl_r2 = r2_score(y_test, dl_predictions)

print("\n>>> [6/6] MODEL EVALUATION & STRATEGIC INSIGHTS")
print("-" * 60)
print(f"[METRIC] Random Forest R2 Score : {rf_r2:.4f}")
print(f"[METRIC] Neural Network R2 Score: {dl_r2:.4f}")
print("-" * 60)

future_scenario = [[30, le_gender.transform(['Female'])[0], le_category.transform(['Smartphone'])[0], 11, 2]]
future_scaled = scaler_ml.transform(future_scenario)

predicted_revenue = rf_model.predict(future_scaled)[0]

print("\n[EXECUTIVE SUMMARY FOR OPERATION DIRECTOR]")
print(f"[*] Scenario: Female, Age 30, Q4 Smartphone Purchase (Qty: 2)")
print(f"[*] AI Predicted Revenue: ${predicted_revenue:,.2f}")
print("\n[STRATEGIC RECOMMENDATIONS]")
print("1. SUPPLY CHAIN OPTIMIZATION")
print("2. LOGISTICS PRIORITIZATION")
print("==============================================================================")