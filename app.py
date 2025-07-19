from flask import Flask, request, render_template
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Loading the dataset, model, encoders, and scaler
df = pd.read_csv('ndtv_data_final.csv')
model = joblib.load('phone_price_model.pkl')
brand_encoder = joblib.load('brand_encoder.pkl')
processor_encoder = joblib.load('processor_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Getting unique brands and processors for dropdowns
brands = sorted(df['Brand'].unique())
processors = sorted(df['Processor'].unique())

@app.route('/')
def home():
    return render_template('index.html', brands=brands, processors=processors)

@app.route('/predict', methods=['POST'])
def predict():
    # Getting user input
    brand = request.form['brand']
    battery = float(request.form['battery'])
    screen_size = float(request.form['screen_size'])
    processor = request.form['processor']
    ram = float(request.form['ram'])
    storage = float(request.form['storage'])
    rear_camera = float(request.form['rear_camera'])
    front_camera = float(request.form['front_camera'])

    # Preparing input data
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Battery capacity (mAh)': [battery],
        'Screen size (inches)': [screen_size],
        'Processor': [processor],
        'RAM (MB)': [ram * 1000],  # Convert GB to MB
        'Internal storage (GB)': [storage],
        'Rear camera': [rear_camera],
        'Front camera': [front_camera]
    })

    # Encoding categorical variables
    input_data['Brand'] = brand_encoder.transform(input_data['Brand'])
    input_data['Processor'] = processor_encoder.transform(input_data['Processor'])

    # Normalizing numerical features
    numerical_features = ['Battery capacity (mAh)', 'Screen size (inches)', 'RAM (MB)', 
                          'Internal storage (GB)', 'Rear camera', 'Front camera']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Predicting price
    predicted_price = model.predict(input_data)[0]

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

    return render_template(
        'result.html',
        predicted_price=round(predicted_price, 2),
        bar_chart=bar_json,
        radar_chart=radar_json
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)