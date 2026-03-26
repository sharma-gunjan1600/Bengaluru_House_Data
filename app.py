from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        location = data["location"]
        total_sqft = float(data["total_sqft"])
        bath = int(data["bath"])
        Bhk = int(data["Bhk"])   # ✅ EXACT column name

        # ✅ Convert to DataFrame because model expects column names
        input_df = pd.DataFrame([{
            "location": location,
            "total_sqft": total_sqft,
            "bath": bath,
            "Bhk": Bhk
        }])

        predicted_price = model.predict(input_df)[0]

        return jsonify({"estimated_price_lakhs": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
