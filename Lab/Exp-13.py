#exp-13
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Kilometers': [10000, 20000, 30000, 40000, 50000,
                   60000, 70000, 80000, 90000, 100000],
    'Price': [900000, 820000, 750000, 680000, 600000,
              530000, 460000, 400000, 330000, 280000]
}

df = pd.DataFrame(data)

X = df[['Age', 'Kilometers']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Car Price Prediction")
print("--------------------")
print("Actual Prices:", list(y_test))
print("Predicted Prices:", list(y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# New car prediction
new_car = [[3, 30000]]

print("Predicted Price:",
      model.predict(new_car)[0])
