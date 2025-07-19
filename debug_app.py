from flask import Flask, request, render_template
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json
import traceback

app = Flask(__name__)

# Loading the dataset, model, encoders, and scaler with error handling
try:
    print("Loading dataset...")
    df = pd.read_csv('ndtv_data_final.csv')
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = None

try:
    print("Loading model...")
    model = joblib.load('phone_price_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    print("Loading encoders...")
    brand_encoder = joblib.load('brand_encoder.pkl')
    processor_encoder = joblib.load('processor_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Encoders loaded successfully")
except Exception as e:
    print(f"Error loading encoders: {e}")
    brand_encoder = None
    processor_encoder = None
    scaler = None
    feature_names = None

# Getting unique brands and processors for dropdowns
if df is not None:
    brands = sorted(df['Brand'].unique())
    processors = sorted(df['Processor'].unique())
else:
    brands = []
    processors = []

@app.route('/')
def home():
    return render_template('index.html', brands=brands, processors=processors)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request")
        
        # Getting user input
        brand = request.form['brand']
        battery = float(request.form['battery'])
        screen_size = float(request.form['screen_size'])
        processor = request.form['processor']
        ram = float(request.form['ram'])
        storage = float(request.form['storage'])
        rear_camera = float(request.form['rear_camera'])
        front_camera = float(request.form['front_camera'])
        
        print(f"Input received: Brand={brand}, Battery={battery}, Screen={screen_size}, Processor={processor}, RAM={ram}, Storage={storage}, Rear={rear_camera}, Front={front_camera}")

        # Creating engineered features
        camera_total = rear_camera + front_camera
        ram_gb = ram
        ram_mb = ram * 1000
        
        # Calculate price per GB and MP for feature engineering
        price_per_gb = df['Price'].mean() / df['Internal storage (GB)'].mean()
        price_per_mp = df['Price'].mean() / (df['Rear camera'].mean() + df['Front camera'].mean())
        screen_to_battery_ratio = screen_size / (battery / 1000)

        # Preparing input data with all features
        input_data = pd.DataFrame({
            'Brand': [brand],
            'Battery capacity (mAh)': [battery],
            'Screen size (inches)': [screen_size],
            'Processor': [processor],
            'RAM (MB)': [ram_mb],
            'Internal storage (GB)': [storage],
            'Rear camera': [rear_camera],
            'Front camera': [front_camera],
            'Camera_Total': [camera_total],
            'RAM_GB': [ram_gb],
            'Price_per_GB': [price_per_gb],
            'Price_per_MP': [price_per_mp],
            'Screen_to_Battery_Ratio': [screen_to_battery_ratio],
            'Price_per_RAM': [price_per_gb],  # Using price_per_gb as approximation
            'Battery_to_Screen_Ratio': [battery / screen_size]
        })

        print("Input data prepared")

        # Encoding categorical variables
        input_data['Brand'] = brand_encoder.transform(input_data['Brand'])
        input_data['Processor'] = processor_encoder.transform(input_data['Processor'])
        
        print("Categorical variables encoded")

        # Scaling all features
        input_scaled = scaler.transform(input_data)
        
        print("Features scaled")

        # Predicting price
        predicted_price = model.predict(input_scaled)[0]
        
        print(f"Prediction made: {predicted_price}")

        # Finding similar phones (based on storage and RAM)
        similar_phones = df[
            (df['Internal storage (GB)'].between(storage * 0.8, storage * 1.2)) &
            (df['RAM (MB)'].between(ram * 1000 * 0.8, ram * 1000 * 1.2))
        ]
        avg_price = similar_phones['Price'].mean() if not similar_phones.empty else predicted_price

        # Creating bar chart
        bar_fig = px.bar(
            x=['Predicted Price', 'Average Price of Similar Phones'],
            y=[predicted_price, avg_price],
            labels={'x': 'Price Type', 'y': 'Price (₹)'},
            title='Price Comparison'
        )
        bar_json = json.dumps(bar_fig, cls=px.utils.PlotlyJSONEncoder)

        # Creating radar chart for feature comparison
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=[battery / 5000, screen_size / 7, ram / 8, storage / 256, rear_camera / 64, front_camera / 32],
            theta=['Battery', 'Screen Size', 'RAM', 'Storage', 'Rear Camera', 'Front Camera'],
            fill='toself',
            name='Input Phone'
        ))
        radar_fig.add_trace(go.Scatterpolar(
            r=[
                df['Battery capacity (mAh)'].mean() / 5000,
                df['Screen size (inches)'].mean() / 7,
                df['RAM (MB)'].mean() / 8000,
                df['Internal storage (GB)'].mean() / 256,
                df['Rear camera'].mean() / 64,
                df['Front camera'].mean() / 32
            ],
            theta=['Battery', 'Screen Size', 'RAM', 'Storage', 'Rear Camera', 'Front Camera'],
            fill='toself',
            name='Average Phone'
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Feature Comparison'
        )
        radar_json = json.dumps(radar_fig, cls=px.utils.PlotlyJSONEncoder)

        print("Charts created successfully")

        return render_template(
            'result.html',
            predicted_price=round(predicted_price, 2),
            bar_chart=bar_json,
            radar_chart=radar_json
        )
        
    except Exception as e:
        print(f"Error in predict function: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 