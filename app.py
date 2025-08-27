from flask import Flask, render_template, request
import pandas as pd 
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
     
    try:
       
        features = {
            "longitude": float(request.form['longitude']),
            "latitude": float(request.form['latitude']),
            "housing_median_age": float(request.form['housing_median_age']),
            "total_rooms": float(request.form['total_rooms']),
            "total_bedrooms": float(request.form['total_bedrooms']),
            "population": float(request.form['population']),
            "households": float(request.form['households']),
            "median_income": float(request.form['median_income']),
            "ocean_proximity": request.form['ocean_proximity']
        }

        input_df = pd.DataFrame([features])

        transformed_input = pipeline.transform(input_df)
        prediction = model.predict(transformed_input)[0]
        prediction = round(prediction, 2)
        return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
    


if __name__ == "__main__":
    app.run(debug=True)
  

