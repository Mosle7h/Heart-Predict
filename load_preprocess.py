# import numpy as np
# import pandas as pd

# # Load the heart disease data
# heart_data = pd.read_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/heart_disease_data.csv")

# # Load the model metrics data
# metrics_data = pd.read_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/model_metrics.csv")

# # Separate features (X) and target label (y)
# X = heart_data.drop("HeartDisease", axis=1)
# y = heart_data["HeartDisease"]

# from sklearn.preprocessing import LabelEncoder

# # Encode categorical columns
# categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
# X_encoded = X.copy()
# encoder = LabelEncoder()

# for col in categorical_cols:
#     X_encoded[col] = encoder.fit_transform(X[col])

# from sklearn.model_selection import train_test_split

# # Perform a train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# from sklearn.preprocessing import StandardScaler

# # Initialize the scaler
# scaler = StandardScaler()

# # Fit the scaler on training data and transform both training and testing data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Save to CSV
# pd.DataFrame(X_train_scaled).to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/X_train_scaled.csv", index=False, encoding='utf-8')
# pd.DataFrame(X_test_scaled).to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/X_test_scaled.csv", index=False, encoding='utf-8')
# y_train.to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/y_train.csv", index=False, encoding='utf-8')
# y_test.to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/y_test.csv", index=False, encoding='utf-8')


# # Check for missing values
# print("Missing values in training data:\n", X_train.isnull().sum())
# print("Missing values in testing data:\n", X_test.isnull().sum())

# # If there are any missing values, fill them or drop rows/columns as needed
# # Example: Fill missing values with mean or median (if any)
# X_train.fillna(X_train.mean(), inplace=True)
# X_test.fillna(X_test.mean(), inplace=True)


# from imblearn.over_sampling import SMOTE

# # Apply SMOTE to the training data only
# smote = SMOTE(random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)


# # Finalize data variables for training and testing
# # X_train_balanced, y_train_balanced, X_test_scaled, y_test

# # Optionally, save the balanced data for future use
# pd.DataFrame(X_train_balanced).to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/X_train_balanced.csv", index=False, encoding='utf-8')
# y_train_balanced.to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/y_train_balanced.csv", index=False, encoding='utf-8')
# pd.DataFrame(X_test_scaled).to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/X_test_scaled.csv", index=False, encoding='utf-8')
# y_test.to_csv("C:/Users/User/Desktop/Internship/Heart-Disease-Prediction/Heart-Disease-Prediction/data/y_test.csv", index=False, encoding='utf-8')



# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# # Model architecture with corrected input shape
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(11,)),  # Corrected input shape syntax
#     Dropout(0.3),
#     Dense(32, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='sigmoid')  # Sigmoid for binary classification
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     X_train_balanced, y_train_balanced,
#     epochs=50,
#     batch_size=32,
#     validation_split=0.2,
#     verbose=1
# )

# # Verify and reshape data if needed to ensure 11 features per sample
# X_train_balanced = np.reshape(X_train_balanced, (-1, 11))
# X_test_scaled = np.reshape(X_test_scaled, (-1, 11))

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
# # Use this format for compatibility with Windows
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%".encode('utf-8', 'ignore').decode('utf-8'))

# import matplotlib.pyplot as plt

# # Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# print("Shape of X_train_balanced:", X_train_balanced.shape)
# print("Shape of X_test_scaled:", X_test_scaled.shape)
# print("Shape of X_test_scaled before evaluation:", X_test_scaled.shape)  # Should be (samples, 11)




# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)



# import sys
# sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+

# # Save the trained model to a .h5 file
# model.save("saved_models/heart_disease_predictor.h5")
# print("Model saved to 'saved_models/heart_disease_predictor.h5'")








import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

# Create directory for saving models if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Load the heart disease data
heart_data = pd.read_csv("data/heart_disease_data.csv")

# Separate features (X) and target label (y)
X = heart_data.drop("HeartDisease", axis=1)
y = heart_data["HeartDisease"]

# Debug print to verify training labels
print("Unique labels in y:", y.unique())

# Encode categorical columns
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    encoders[col] = encoder

# Save the encoders
for col, encoder in encoders.items():
    joblib.dump(encoder, f'saved_models/{col}_encoder.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debug print to verify training labels after split
print("Unique labels in y_train after split:", y_train.unique())
print("Unique labels in y_test after split:", y_test.unique())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'saved_models/scaler.pkl')

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Debug print to verify labels after SMOTE
print("Unique labels in y_train_balanced after SMOTE:", y_train_balanced.unique())

# Define model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_balanced, y_train_balanced,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save the model
model.save("saved_model/heart_disease_predictor.h5")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
