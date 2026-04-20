import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import streamlit as st
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("chennai_flood_rainfall_data.tsv", sep='\t')

# Preprocess data
data.fillna(method='ffill', inplace=True)
X = data[['Temperature', 'Humidity', 'Wind Speed', 'Pressure', 'Rainfall', 'Cloud Cover', 'Air Pollution']]

# Encode 'Event' labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Event'])  
joblib.dump(label_encoder, 'label_encoder.pkl')  # Save label encoder

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')  # Save scaler

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgb_model.pkl')

# Train Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'dt_model.pkl')

# Train ensemble model
ensemble_model = VotingClassifier(estimators=[('xgb', xgb_model), ('dt', dt_model)], voting='hard')
ensemble_model.fit(X_train, y_train)
joblib.dump(ensemble_model, 'ensemble_model.pkl')

# Load saved models for Streamlit
xgb_model = joblib.load('xgb_model.pkl')
dt_model = joblib.load('dt_model.pkl')
ensemble_model = joblib.load('ensemble_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("🌧️ Chennai Flood & Heavy Rainfall Prediction Dashboard")

# User Inputs
temp = st.slider("Temperature (°C)", float(data['Temperature'].min()), float(data['Temperature'].max()), float(data['Temperature'].mean()))
humidity = st.slider("Humidity (%)", float(data['Humidity'].min()), float(data['Humidity'].max()), float(data['Humidity'].mean()))
wind_speed = st.slider("Wind Speed (km/h)", float(data['Wind Speed'].min()), float(data['Wind Speed'].max()), float(data['Wind Speed'].mean()))
pressure = st.slider("Pressure (hPa)", float(data['Pressure'].min()), float(data['Pressure'].max()), float(data['Pressure'].mean()))
rainfall = st.slider("Rainfall (mm)", float(data['Rainfall'].min()), float(data['Rainfall'].max()), float(data['Rainfall'].mean()))
cloud_cover = st.slider("Cloud Cover (%)", float(data['Cloud Cover'].min()), float(data['Cloud Cover'].max()), float(data['Cloud Cover'].mean()))
air_pollution = st.slider("Air Pollution Index", float(data['Air Pollution'].min()), float(data['Air Pollution'].max()), float(data['Air Pollution'].mean()))

# Prepare user input for prediction
user_input = np.array([[temp, humidity, wind_speed, pressure, rainfall, cloud_cover, air_pollution]])
user_input_scaled = scaler.transform(user_input)

# Predict event
prediction = ensemble_model.predict(user_input_scaled)[0]
prediction_text = label_encoder.inverse_transform([prediction])[0]  

st.subheader(f"🌟 Prediction: {prediction_text}")

# Map Visualization
st.subheader("🗺️ Chennai Risk Map")
map_center = [13.0827, 80.2707]  # Chennai coordinates
m = folium.Map(location=map_center, zoom_start=10)
folium.Marker(map_center, popup=prediction_text, icon=folium.Icon(color="red")).add_to(m)
folium_static(m)

# Historical Data Graph
st.subheader("📊 Historical Rainfall & Flood Events")
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Rainfall'], label="Rainfall (mm)", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Rainfall (mm)", color="blue")
ax2 = ax.twinx()
ax2.plot(data['Date'], label_encoder.transform(data['Event']), label="Event Type", color="red", linestyle="dashed")
ax2.set_ylabel("Event Type", color="red")
st.pyplot(fig)

st.write("✅ **Dashboard successfully integrated with predictive models!**")
