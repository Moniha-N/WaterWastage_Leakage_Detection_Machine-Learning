import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("waterleakagedataset.csv")
data.columns = data.columns.str.strip()

# Rename column to avoid space issues
data.rename(columns={'User Present': 'User_Present'}, inplace=True)
print("Columns:", data.columns)
print(data.head())

# Convert Time to numeric
data['Time'] = data['Time'].map({
    'Morning': 0,
    'Afternoon': 1,
    'Evening': 2,
    'Night': 3
})

# Convert User_Present (Low/High → 0/1)
data['User_Present'] = data['User_Present'].map({
    'Low': 0,
    'High': 1
})

X = data[['Tap Duration', 'Frequency', 'Time', 'User_Present']]
y = data['Wastage']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure()
plt.plot(y_test.values, marker='o', label='Actual')
plt.plot(y_pred, marker='x', label='Predicted')
plt.title("Actual vs Predicted Wastage")
plt.xlabel("Test Samples")
plt.ylabel("Wastage (0/1)")
plt.legend()
plt.show()

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.colorbar()
plt.show()



