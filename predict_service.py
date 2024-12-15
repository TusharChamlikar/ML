from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load(r"C:\Users\chaml\OneDrive\Documents\finalproject\fortest\rentalsystem.pkl")


# Load the encoders
le_house_type = joblib.load("le_house_type1.pkl")
le_location = joblib.load("le_location1.pkl")
le_city = joblib.load("le_city1.pkl")
le_amenities = joblib.load("le_amenities1.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    # Prepare the input data for prediction
    input_data = {
        'house_type': [data.get('house_type')],
        'house_size': [float(data.get('house_size'))],
        'location': [data.get('location')],
        'city': [data.get('city')],
        'amenities': [data.get('amenities')]
    }
    input_df = pd.DataFrame(input_data)

    # Encode the categorical variables
    input_df['house_type'] = le_house_type.transform(input_df['house_type'])
    input_df['location'] = le_location.transform(input_df['location'])
    input_df['city'] = le_city.transform(input_df['city'])
    input_df['amenities'] = le_amenities.transform(input_df['amenities'])

    # Predict the price
    predicted_price = model.predict(input_df)[0]

    return jsonify({"predicted_price": predicted_price})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

