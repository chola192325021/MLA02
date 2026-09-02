#exp-19
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
data = {
    'Age': [25, 35, 45, 28, 50, 40, 30, 55, 32, 48],
    'Income': [25000, 45000, 70000, 30000, 80000,
               60000, 35000, 90000, 40000, 75000],
    'LoanAmount': [10000, 20000, 30000, 12000, 40000,
                   25000, 15000, 50000, 18000, 35000],
    'Loan_Status': ['No', 'Yes', 'Yes', 'No', 'Yes',
                    'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Encode target
encoder = LabelEncoder()
y = encoder.fit_transform(df['Loan_Status'])

X = df[['Age', 'Income', 'LoanAmount']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Bank Loan Prediction")
print("--------------------")

print("Accuracy:",
      accuracy_score(y_test, y_pred) * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# New customer
new_customer = [[35, 50000, 20000]]

prediction = model.predict(new_customer)

print("\nLoan Prediction:",
      encoder.inverse_transform(prediction)[0])
