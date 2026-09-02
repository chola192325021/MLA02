#exp-17
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'RAM': [2, 3, 4, 6, 8, 2, 4, 6, 8, 12],
    'Storage': [32, 64, 64, 128, 256, 32, 128, 256, 512, 512],
    'Battery': [3000, 3500, 4000, 4500, 5000,
                3000, 4500, 5000, 5000, 6000],
    'Price_Range': ['Low', 'Low', 'Medium', 'Medium', 'High',
                    'Low', 'Medium', 'High', 'High', 'High']
}

df = pd.DataFrame(data)

X = df[['RAM', 'Storage', 'Battery']]
y = df['Price_Range']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Mobile Price Prediction")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# New mobile
new_mobile = [[6, 128, 4500]]
prediction = model.predict(new_mobile)

print("Predicted Price Range:", prediction[0])
