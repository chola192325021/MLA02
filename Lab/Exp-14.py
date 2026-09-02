#exp-14
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'Area': [800, 1000, 1200, 1400, 1600,
             1800, 2000, 2200, 2400, 2600],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'Price': [2000000, 2500000, 3000000, 3500000, 4000000,
              4500000, 5000000, 5500000, 6000000, 6500000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("House Price Prediction")
print("---------------------")
print("Actual Prices:", list(y_test))
print("Predicted Prices:", list(y_pred))

print("R2 Score:", r2_score(y_test, y_pred))

# New house
new_house = [[1500, 3]]

print("Predicted House Price:",
      model.predict(new_house)[0])
