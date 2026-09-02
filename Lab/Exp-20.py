#exp-20
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sales data
months = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10
]).reshape(-1, 1)

sales = np.array([
    100, 120, 140, 160, 180,
    200, 220, 240, 260, 280
])

# Create model
model = LinearRegression()

# Train model
model.fit(months, sales)

# Predict future sales
future_months = np.array([
    11, 12, 13, 14, 15
]).reshape(-1, 1)

future_sales = model.predict(future_months)

print("Future Sales Prediction")
print("-----------------------")

for month, sale in zip(future_months.flatten(), future_sales):
    print("Month", month, "Predicted Sales:", round(sale, 2))

# Plot
plt.scatter(months, sales, label="Actual Sales")
plt.plot(months, model.predict(months), label="Regression Line")
plt.plot(future_months, future_sales, label="Future Prediction")

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Future Sales Prediction")
plt.legend()
plt.show()
