from flask import Flask, render_template, request
import joblib
import pandas as pd
import csv
import os

app = Flask(
    __name__,
    template_folder= "../Frontend/templates",
    static_folder= "../Frontend/static"
)

model =joblib.load("../ML/house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html",form_data=None)

@app.route("/predict", methods=["POST"])
def predict():
    area = int(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    stories = int(request.form["stories"])
    mainroad = int(request.form["mainroad"])
    guestroom = int(request.form["guestroom"])
    basement = int(request.form["basement"])
    hotwaterheating = int(request.form["hotwaterheating"])
    airconditioning = int(request.form["airconditioning"])
    parking = int(request.form["parking"])
    prefarea = int(request.form["prefarea"])
    furnishingstatus = request.form["furnishingstatus"]
    
    furnishingstatus_1 = 1 if furnishingstatus == "semi-furnished" else 0
    furnishingstatus_2 =1 if furnishingstatus == "unfurnished" else 0

    input_data = pd.DataFrame([[
        area, bedrooms, bathrooms, stories, mainroad, guestroom,
        basement, hotwaterheating, airconditioning, parking, prefarea,
        furnishingstatus_1, furnishingstatus_2
    ]], columns=[
        'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
        'basement', 'hotwaterheating', 'airconditioning', 'parking',
        'prefarea', 'furnishingstatus_1', 'furnishingstatus_2'
    ])

    prediction = model.predict(input_data)[0]

    history_file = "prediction_history.csv"

    file_exists = os.path.exists(history_file)

    with open(history_file, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Area", "Bedrooms", "Bathrooms", "Stories",
                "Mainroad", "Guestroom", "Basement",
                "Hotwaterheating", "Airconditioning",
                "Parking", "Prefarea", "Furnishingstatus",
                "Predicted Price"
            ])

        writer.writerow([
            area, bedrooms, bathrooms, stories,
            mainroad, guestroom, basement,
            hotwaterheating, airconditioning,
            parking, prefarea, furnishingstatus,
            prediction
        ])

    form_data = request.form

    return render_template(
     "index.html",
        prediction = f"Estimated House Price: ₹{prediction:,.2f}",
        form_data = form_data
    )

@app.route("/history")
def history():
    records = []

    if os.path.exists("prediction_history.csv"):
        with open("prediction_history.csv", "r") as file:
            reader = csv.reader(file)
            records = list(reader)

    return render_template("history.html", records=records)

if __name__ == "__main__":
    app.run(debug=False)
