import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('ford.csv')

# print(df.head())
# print(df.info())
# print(df.isnull().sum())

# sns.histplot(df['price'], bins=50, kde=True)
# print(df.corr(numeric_only=True))
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# sns.boxplot(data=df, x='transmission', y='price')
# sns.boxplot(data=df, x='fuelType', y='price')

X = df.drop(columns=['price'])
y = df['price']

X_one_encoded = pd.get_dummies(
    X,
    columns=['model', 'transmission', 'fuelType'],
    drop_first=True
)

X_one_encoded = X_one_encoded.astype(int)

label_encoder = LabelEncoder()

X_label_encoded = X.copy()

for col in ['model', 'transmission', 'fuelType']:
    X_label_encoded[col] = label_encoder.fit_transform(X_label_encoded[col])

numeric_cols = ['year','mileage','tax','mpg','engineSize']
scaler = StandardScaler()
X_one_encoded[numeric_cols] = scaler.fit_transform(X_one_encoded[numeric_cols])
X_label_encoded[['year','mileage','tax','mpg','engineSize','model','transmission','fuelType']] = scaler.fit_transform(X_label_encoded[['year','mileage','tax','mpg','engineSize','model','transmission','fuelType']])

# print(X_one_encoded.head())
# print(X_label_encoded.head())
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(
    X_label_encoded, y, test_size=0.2, random_state=42
)

model_label = LinearRegression()
model_label.fit(X_train, y_train)

y_pred = model_label.predict(X_test)

r2_label = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2_label = 1 - ((1 - r2_label) * (n - 1) / (n - p - 1))

print("Label Encoded Model")
print("R²:", r2_label)
print("Adjusted R²:", adjusted_r2_label)


# ---------- One-Hot Encoded Model ----------

X_train, X_test, y_train, y_test = train_test_split(
    X_one_encoded, y, test_size=0.2, random_state=42
)

model_one = LinearRegression()
model_one.fit(X_train, y_train)

y_pred = model_one.predict(X_test)

r2_one = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2_one = 1 - ((1 - r2_one) * (n - 1) / (n - p - 1))

print("\nOne-Hot Encoded Model")
print("R²:", r2_one)
print("Adjusted R²:", adjusted_r2_one)