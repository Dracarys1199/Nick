import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load dataset (TSV format)
data = pd.read_csv("chennai_flood_rainfall_data.tsv", sep='\t')  # Load TSV file

# Preprocess data
data.fillna(method='ffill', inplace=True)
X = data[['Temperature', 'Humidity', 'Wind Speed', 'Pressure', 'Rainfall', 'Cloud Cover', 'Air Pollution']]

# Encode 'Event' column to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Event'])  

# Save label encoder for later
joblib.dump(label_encoder, 'label_encoder.pkl')

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')  # Save scaler

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Models
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgb_model.pkl')

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'dt_model.pkl')

# Train an ensemble model
ensemble_model = VotingClassifier(estimators=[('xgb', xgb_model), ('dt', dt_model)], voting='hard')
ensemble_model.fit(X_train, y_train)
joblib.dump(ensemble_model, 'ensemble_model.pkl')
