#exp-11
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'Income': [25000, 40000, 60000, 30000, 80000, 50000, 70000, 20000, 90000, 45000],
    'Loan': [10000, 15000, 20000, 12000, 25000, 18000, 22000, 8000, 30000, 16000],
    'CreditScore': ['Low', 'Low', 'High', 'Low', 'High',
                    'High', 'High', 'Low', 'High', 'High']
}

df = pd.DataFrame(data)

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['CreditScore'])

X = df[['Income', 'Loan']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Credit Score Classification")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# New customer
new_customer = [[55000, 17000]]
prediction = model.predict(new_customer)

print("Predicted Credit Score:",
      le.inverse_transform(prediction)[0])
