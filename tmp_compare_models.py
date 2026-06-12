import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('clarity_bookings_dataset.csv')
df['booking_date'] = pd.to_datetime(df['booking_date'], errors='coerce')
df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')
df['target'] = df['booking_status'].isin(['Cancelled', 'Refunded']).astype(int)
df = df.drop_duplicates(subset=['booking_id'], keep='first').copy()
for col in ['payment_method', 'fare_basis', 'booking_channel', 'booking_source', 'airline', 'origin', 'destination']:
    df[col] = df[col].fillna('Unknown')
df['customer_complaint'] = df['customer_complaint'].fillna('No')
df['satisfaction_score'] = df['satisfaction_score'].fillna(df['satisfaction_score'].median())
df['ancillary_revenue_inr'] = df['ancillary_revenue_inr'].fillna(0)
df['trip_duration_days'] = (df['departure_date'] - df['booking_date']).dt.days.clip(lower=0)
route_avg = df.groupby(['origin', 'destination'])['total_fare_inr'].mean().rename('avg_route_fare')
df = df.merge(route_avg, on=['origin', 'destination'], how='left')
df['fare_per_passenger'] = df['total_fare_inr'] / df['pax_count'].replace(0, np.nan)
df['ancillary_per_passenger'] = df['ancillary_revenue_inr'] / df['pax_count'].replace(0, np.nan)
df['fare_to_route_avg'] = df['total_fare_inr'] / df['avg_route_fare'].replace(0, np.nan)
df['booking_month'] = df['booking_date'].dt.month
df['is_repeat_customer'] = (df['prior_bookings'] > 0).astype(int)
df['has_customer_complaint'] = df['customer_complaint'].astype(str).str.contains('Yes|yes', regex=True).astype(int)
df['lead_time_bucket'] = pd.cut(df['lead_time_days'], bins=[-1, 7, 30, 60, 90, 365], labels=['0-7', '8-30', '31-60', '61-90', '90+'])
df['fare_to_income_proxy'] = df['total_fare_inr'] / (df['pax_count'] + 1)
for col in ['fare_per_passenger', 'ancillary_per_passenger', 'fare_to_route_avg', 'fare_to_income_proxy']:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
feature_cols = ['lead_time_days','pax_count','base_fare_inr','taxes_inr','total_fare_inr','ancillary_revenue_inr','prior_bookings','satisfaction_score','fare_per_passenger','ancillary_per_passenger','fare_to_route_avg','booking_month','trip_duration_days','fare_to_income_proxy','haul_type','cabin_class','trip_type','fare_basis','booking_channel','booking_source','payment_method','airline','origin','destination','lead_time_bucket','is_repeat_customer','has_customer_complaint']
X = df[feature_cols].copy()
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
numeric = X.select_dtypes(include=['number']).columns.tolist()
categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
], remainder='drop')
models = [
    ('LogReg', Pipeline([('pre', preprocessor), ('m', LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear'))])),
    ('RF', Pipeline([('pre', preprocessor), ('m', RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42, n_jobs=-1))])),
    ('ExtraTrees', Pipeline([('pre', preprocessor), ('m', ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1))]))
]
for name, model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(name, 'P=', round(precision_score(y_test, pred, zero_division=0), 3), 'R=', round(recall_score(y_test, pred, zero_division=0), 3), 'F1=', round(f1_score(y_test, pred, zero_division=0), 3), 'AUC=', round(roc_auc_score(y_test, proba), 3))
