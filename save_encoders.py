import joblib
from sklearn.preprocessing import LabelEncoder

# Assume encoder fitting happens here
encoders = {}
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for col in categorical_cols:
    encoder = LabelEncoder()
    X_encoded[col] = encoder.fit_transform(X[col])
    encoders[col] = encoder

# Save the encoders
for col, encoder in encoders.items():
    joblib.dump(encoder, f'saved_models/{col}_encoder.pkl')
