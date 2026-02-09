#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "C:/Users/Anarghya/OneDrive/Desktop/major project/energy_consumption_dataset in csv.csv" 
df = pd.read_csv(file_path)

# Encode categorical variable (Weather)
df["Weather"] = LabelEncoder().fit_transform(df["Weather"])

# Select features and target
X = df.drop(columns=["Total Energy Consumption (kWh)"])
y = df["Total Energy Consumption (kWh)"]

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape for LSTM (samples, time steps, features)
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42, shuffle=True)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# Predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_inv = scaler_y.inverse_transform(y_test)

# Display Prediction
print(y_pred)

# Evaluation Metrics
mse = mean_squared_error(y_test_inv, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred)

print(f"\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

# Plot Predictions vs Actual
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x')
plt.xlabel("Sample")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.title("Actual vs Predicted Energy Consumption")
plt.show()


# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Define energy consumption categories
def categorize_energy(value, low_thresh, high_thresh):
    if value < low_thresh:
        return "Low"
    elif value < high_thresh:
        return "Medium"
    else:
        return "High"

# Define thresholds based on percentiles
low_threshold = np.percentile(y_test_inv, 33)
high_threshold = np.percentile(y_test_inv, 66)

# Convert actual and predicted values to categories
y_test_class = np.array([categorize_energy(val, low_threshold, high_threshold) for val in y_test_inv.flatten()])
y_pred_class = np.array([categorize_energy(val, low_threshold, high_threshold) for val in y_pred.flatten()])

# Create confusion matrix
conf_matrix = confusion_matrix(y_test_class, y_pred_class, labels=["Low", "Medium", "High"])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Energy Consumption Categories")
plt.show()

# Print classification report
print(classification_report(y_test_class, y_pred_class))


# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

# Load dataset
file_path = "C:/Users/Anarghya/OneDrive/Desktop/major project/energy_consumption_dataset in csv.csv"
df = pd.read_csv(file_path)

# Encode categorical variable (if any)
df["Weather"] = df["Weather"].astype("category").cat.codes  # Convert categorical to numeric

# Select features and target
X = df.drop(columns=["Total Energy Consumption (kWh)"])  # Features
y = df["Total Energy Consumption (kWh)"]  # Target variable

# Compute feature importance using Mutual Information
feature_importance = mutual_info_regression(X, y)

# Create a DataFrame to store and sort feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(9, 7))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Load Forecasting")
plt.gca().invert_yaxis()
plt.show()

# Display ranked features
print(feature_importance_df)


# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error


# Load dataset
file_path = "C:/Users/Anarghya/OneDrive/Desktop/major project/energy_consumption_dataset in csv.csv" 
df = pd.read_csv(file_path)

# Encode categorical variable (Weather)
df["Weather"] = LabelEncoder().fit_transform(df["Weather"])

# Select features and target
X = df.drop(columns=["Total Energy Consumption (kWh)"])
y = df["Total Energy Consumption (kWh)"]

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape for LSTM (samples, time steps, features)
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42, shuffle=True)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# Predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_inv = scaler_y.inverse_transform(y_test)

# Display Prediction
print(y_pred)

# Evaluation Metrics
mse = mean_squared_error(y_test_inv, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred)
mae = mean_absolute_error(y_test_inv, y_pred)


print(f"\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()


# In[5]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error


# Load dataset
file_path = "C:/Users/Anarghya/OneDrive/Desktop/major project/energy_consumption_dataset in csv.csv" 
df = pd.read_csv(file_path)

# Encode categorical variable (Weather)
df["Weather"] = LabelEncoder().fit_transform(df["Weather"])

# Select features and target
X = df.drop(columns=["Total Energy Consumption (kWh)"])
y = df["Total Energy Consumption (kWh)"]

# Normalize data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape for LSTM (samples, time steps, features)
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_scaled, test_size=0.2, random_state=42, shuffle=True)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

# Predictions
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_inv = scaler_y.inverse_transform(y_test)

# Display Prediction
print(y_pred)

# Evaluation Metrics
mse = mean_squared_error(y_test_inv, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred)
mae = mean_absolute_error(y_test_inv, y_pred)


print(f"\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}")

mape = np.mean(np.abs((y_test_inv - y_pred) / y_test_inv)) * 100
print("MAPE: ", mape)

residuals = y_test_inv.flatten() - y_pred.flatten()
plt.figure(figsize=(8, 4))
plt.scatter(range(len(residuals)), residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals of Predictions")
plt.xlabel("Sample")
plt.ylabel("Residual (Actual - Predicted)")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_inv, y_pred, alpha=0.7)
plt.plot([min(y_test_inv), max(y_test_inv)], [min(y_test_inv), max(y_test_inv)], 'r--')
plt.xlabel("Actual Energy Consumption (kWh)")
plt.ylabel("Predicted Energy Consumption (kWh)")
plt.title("Actual vs Predicted Energy (Parity Plot)")
plt.grid(True)
plt.show()

model.summary()


# In[4]:


# Plot feature importance
plt.figure(figsize=(9, 8))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Load Forecasting")
plt.gca().invert_yaxis()
plt.show()


# In[ ]:




