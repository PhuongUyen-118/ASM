import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load model và scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📈 Dự Đoán Doanh Số Hàng Tuần Walmart")

st.write("Nhập các giá trị đầu vào:")

temperature = st.slider("🌡️ Nhiệt độ", min_value=0.0, max_value=1.0, value=0.5)
fuel_price = st.slider("⛽ Giá nhiên liệu", min_value=0.0, max_value=1.0, value=0.5)
cpi = st.slider("💹 CPI", min_value=0.0, max_value=1.0, value=0.5)
unemployment = st.slider("📉 Tỷ lệ thất nghiệp", min_value=0.0, max_value=1.0, value=0.5)

# Tạo dataframe đầu vào
input_data = pd.DataFrame([[temperature, fuel_price, cpi, unemployment]],
                          columns=["Temperature", "Fuel_Price", "CPI", "Unemployment"])

# Dự đoán
prediction = model.predict(input_data)[0]

st.subheader("📊 Doanh số dự đoán (chuẩn hóa):")
st.success(f"{prediction:.4f}")
