# Sample Python code for optimizing composite materials using machine learning

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate a synthetic dataset (in practice, you would load real data)
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'Fiber_Content': np.random.uniform(30, 70, n_samples),  # Fiber content (%)
    'Matrix_Type': np.random.choice([0, 1], n_samples),  # Matrix type (binary categorical)
    'Curing_Temperature': np.random.uniform(100, 200, n_samples),  # Curing temperature (°C)
    'Curing_Time': np.random.uniform(1, 5, n_samples),  # Curing time (hours)
    'Tensile_Strength': np.random.uniform(300, 500, n_samples)  # Tensile strength (MPa)
})

# Features and target variable
X = data.drop('Tensile_Strength', axis=1)  # Features
y = data['Tensile_Strength']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Example of predicting tensile strength for a new sample
new_sample = np.array([[50, 1, 150, 3]])  # Example input
new_sample_scaled = scaler.transform(new_sample)
predicted_strength = model.predict(new_sample_scaled)
print(f"Predicted Tensile Strength: {predicted_strength[0]:.2f} MPa")
