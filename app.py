from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('models/price_predictor_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load dataset for categorical encoding reference
df = pd.read_csv('ndtv_data_final.csv')
categorical_columns = ['Name', 'Brand', 'Model', 'Touchscreen', 'Operating system', 
                     'Wi-Fi', 'Bluetooth', 'GPS', '3G', '4G/ LTE']
category_mappings = {col: df[col].astype('category').cat.categories for col in categorical_columns}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'Name': request.form['name'],
                'Brand': request.form['brand'],
                'Model': request.form['model'],
                'Battery capacity (mAh)': float(request.form['battery']),
                'Screen size (inches)': float(request.form['screen_size']),
                'Touchscreen': request.form['touchscreen'],
                'Resolution x': float(request.form['resolution_x']),
                'Resolution y': float(request.form['resolution_y']),
                'Processor': int(request.form['processor']),
                'RAM (MB)': float(request.form['ram']) * 1000,  # Convert GB to MB
                'Internal storage (GB)': float(request.form['storage']),
                'Rear camera': float(request.form['rear_camera']),
                'Front camera': float(request.form['front_camera']),
                'Operating system': request.form['os'],
                'Wi-Fi': request.form['wifi'],
                'Bluetooth': request.form['bluetooth'],
                'GPS': request.form['gps'],
                'Number of SIMs': int(request.form['sims']),
                '3G': request.form['3g'],
                '4G/ LTE': request.form['4g']
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([form_data])

            # Encode categorical variables
            for col in categorical_columns:
                if form_data[col] in category_mappings[col]:
                    input_df[col] = category_mappings[col].get_loc(form_data[col])
                else:
                    input_df[col] = -1  # Handle unknown categories

            # Scale numerical features
            numerical_columns = ['Battery capacity (mAh)', 'Screen size (inches)', 'Resolution x', 
                               'Resolution y', 'Processor', 'RAM (MB)', 'Internal storage (GB)', 
                               'Rear camera', 'Front camera']
            input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

            # Make prediction
            prediction = model.predict(input_df)[0]
            return render_template('index.html', prediction=f"Estimated Price: ₹{prediction:,.2f}")

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)