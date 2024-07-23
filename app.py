import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)

try:
    # Load the scaler and model
    scaler = pickle.load(open('Models/sc_california_housing.pkl', 'rb'))
    model = tf.keras.models.load_model('Models/ann_california_housing.h5')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home_page():
    return render_template('index.html', Result=None)

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Retrieve data from form
        Longitude = float(request.form.get('longitude'))
        Latitude = float(request.form.get('latitude'))
        Housing_Age = float(request.form.get('housing_median_age'))
        Average_rooms = float(request.form.get('average_rooms'))
        Average_bedrooms = float(request.form.get('average_bedrooms'))
        Population = float(request.form.get('population'))
        Houseoccupancy = float(request.form.get('houseoccupancy'))
        Median_income = float(request.form.get('median_income'))

        # Prepare the input data for prediction
        new_data = scaler.transform([[Longitude, Latitude, Housing_Age, Average_rooms,
                                     Average_bedrooms, Population, Houseoccupancy, Median_income]])
        
        # Make prediction
        result = model.predict(new_data)
        return render_template('index.html', Result="{:.2f}".format(result[0][0]))
    else:
        return render_template('index.html', Result=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
