from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
import os
import secrets
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bcrypt import Bcrypt
from functools import wraps

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:12345@127.0.0.1:3306/heartpred'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = secrets.token_hex(16)

# Extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)

# User Model
class User(db.Model):
    __tablename__ = 'Users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Login Required Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
@login_required
def home():
    return render_template('index.html')  # Home page where users can make predictions

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        session['user'] = username
        flash('Sign-up successful! You are now logged in.', 'success')
        return redirect(url_for('home'))

    return render_template('sign_up.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('sign_in.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST', 'GET'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Collect form inputs
            age = int(request.form['age'])
            resting_BP = int(request.form['resting_BP'])
            cholesterol = int(request.form['cholesterol'])
            MaxHR = int(request.form['MaxHR'])
            oldpeak = float(request.form['oldpeak'])
            sex = request.form['sex']
            chest_pain = request.form['chest_pain']
            fasting_bs = request.form['fasting_bs']
            resting_ECG = request.form['resting_ECG']
            ExerciseAngina = request.form['ExerciseAngina']
            ST_Slope = request.form['ST_Slope']
            selected_model = request.form['selected_model']

            # Conversion and encoding logic
            def convert_categorical_variables(sex_, chest_pain_, fasting_bs_, resting_ECG_, ExerciseAngina_, ST_Slope_):
                sex_conversion = {"Male": "M", "Female": "F"}
                chest_pain_conversion = {"Typical Angina Pain": "TA", "Atypical Angina Pain": "ATA", "Non-Anginal Pain": "NAP", "No Chest Pain": "ASY"}
                fasting_bs_conversion = {"Over 120": "1", "120 or Under": "0"}
                resting_ECG_conversion = {"Normal": "Normal", "ST": "ST", "LVH": "LVH"}
                ExerciseAngina_conversion = {"Yes": "Y", "No": "N"}
                ST_Slope_conversion = {"Sloping Upwards": "Up", "Flat": "Flat", "Sloping Downwards": "Down"}
                return (
                    sex_conversion[sex_],
                    chest_pain_conversion[chest_pain_],
                    fasting_bs_conversion[fasting_bs_],
                    resting_ECG_conversion[resting_ECG_],
                    ExerciseAngina_conversion[ExerciseAngina_],
                    ST_Slope_conversion[ST_Slope_]
                )

            sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope = convert_categorical_variables(
                sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope
            )

            # Create new instance for prediction
            new_instance = pd.DataFrame({
                "Age": [age],
                "Sex": [sex],
                "ChestPainType": [chest_pain],
                "RestingBP": [resting_BP],
                "Cholesterol": [cholesterol],
                "FastingBS": [fasting_bs],
                "RestingECG": [resting_ECG],
                "MaxHR": [MaxHR],
                "ExerciseAngina": [ExerciseAngina],
                "Oldpeak": [oldpeak],
                "ST_Slope": [ST_Slope]
            })

            # Base path for models
            base_path = os.path.dirname(os.path.abspath(__file__))

            # Load encoders
            encoders = {}
            categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
            for col in categorical_cols:
                encoder_path = os.path.join(base_path, f'saved_models/{col}_encoder.pkl')
                encoders[col] = joblib.load(encoder_path)
                new_instance[col] = encoders[col].transform(new_instance[col])

            # Load scaler
            scaler_path = os.path.join(base_path, 'saved_models/scaler.pkl')
            scaler = joblib.load(scaler_path)
            new_instance_scaled = scaler.transform(new_instance)

            # Convert to numpy array for TensorFlow model
            to_predict = new_instance_scaled.astype(np.float32)

            # Load models
            rf_model_path = os.path.join(base_path, "saved_models/random_forest_classifier.pkl")
            rf_model = joblib.load(rf_model_path)

            dl_model_path = os.path.join(base_path, "saved_model/heart_disease_predictor.h5")
            dl_model = tf.keras.models.load_model(dl_model_path)

            # Prediction logic
            def make_prediction():
                if selected_model == "Random Forest Classifier (Highest Specificity)":
                    prediction = rf_model.predict(to_predict)
                    probability_positive = rf_model.predict_proba(to_predict)[0][1]
                else:
                    predictions = dl_model.predict(to_predict)
                    prediction = np.round(predictions).astype(int)[0]
                    probability_positive = predictions[0][0]

                if prediction[0] == 1:
                    return f"It is predicted that the patient has heart disease.<br>Chance of being heart disease positive: {100 * probability_positive:.2f}%"
                else:
                    return f"It is predicted that the patient does not have heart disease. Chance of being heart disease positive: {100 * probability_positive:.2f}%"

            prediction_result = make_prediction()
            return render_template('result.html', result=prediction_result)

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
