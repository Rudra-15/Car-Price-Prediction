import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("car data.csv")

# Display basic info
print(df.head())
print(df.info())
print(df.describe())

# Checking for missing values
print("Missing Values:\n", df.isnull().sum())

# Visualizing correlations
sns.pairplot(df)
plt.show()

# Encoding categorical features
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
numerical_features = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']

ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_features),
        ('cat', ohe, categorical_features)
    ]
)

# Splitting dataset into features and target
X = df.drop(columns=['Car_Name', 'Selling_Price'])
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Training the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, "car_price_model.pkl")
print("Model saved successfully.")
