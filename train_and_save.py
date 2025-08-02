import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Đọc dữ liệu
df = pd.read_csv("WalmartSales.csv.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')

# Chuẩn hóa
scaler = MinMaxScaler()
cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
df[cols] = scaler.fit_transform(df[cols])

X = df[["Temperature", "Fuel_Price", "CPI", "Unemployment"]]
y = df["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Lưu model và scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
