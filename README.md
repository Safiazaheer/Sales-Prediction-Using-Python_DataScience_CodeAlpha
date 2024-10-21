This project focuses on predicting sales for a given product or company based on historical data and several other features. 
The aim is to develop a machine learning model that can accurately forecast future sales, which can assist businesses in decision-making processes like inventory management, pricing strategies, and marketing campaigns.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv('C:/Users/PMLS/Downloads/Advertising.csv')
# Display the first few rows of the dataset
print(df.head())

# Display the info of the DataFrame
print(df.info())

# Check for missing values
print(df.isnull().sum())
df_clean = df.dropna()
X = df_clean.drop('Sales', axis=1)  # Features
y = df_clean['Sales']  # Target variable
# Example: One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line for reference
plt.show()
