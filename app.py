from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model
model = joblib.load("linear.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>

    <style>

        *{
            margin:0;
            padding:0;
            box-sizing:border-box;
            font-family:Arial, sans-serif;
        }

        body{
            min-height:100vh;
            display:flex;
            justify-content:center;
            align-items:center;

            background:
            linear-gradient(135deg,#667eea,#764ba2);
            padding:20px;
        }

        .container{

            width:100%;
            max-width:950px;

            background:rgba(255,255,255,0.95);

            border-radius:25px;

            box-shadow:
            0 20px 50px rgba(0,0,0,0.25);

            overflow:hidden;
        }

        .header{

            background:
            linear-gradient(135deg,#4f46e5,#7c3aed);

            color:white;
            text-align:center;
            padding:35px;
        }

        .header h1{
            font-size:34px;
        }

        .header p{
            margin-top:10px;
            opacity:0.9;
        }

        .form-area{
            padding:35px;
        }

        .grid{
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:20px;
        }

        .field{
            display:flex;
            flex-direction:column;
        }

        label{
            margin-bottom:8px;
            font-weight:600;
            color:#374151;
        }

        input, select{

            padding:15px;
            border:none;
            border-radius:12px;

            background:#f3f4f6;

            font-size:15px;

            box-shadow:
            inset 0 2px 5px rgba(0,0,0,0.05);

            transition:0.3s;
        }

        input:focus,
        select:focus{

            outline:none;

            box-shadow:
            0 0 0 4px rgba(99,102,241,0.2);
        }

        .full{
            grid-column:1/3;
        }

        button{

            width:100%;
            margin-top:30px;
            padding:16px;

            border:none;
            border-radius:15px;

            background:
            linear-gradient(135deg,#4f46e5,#7c3aed);

            color:white;
            font-size:18px;
            font-weight:bold;

            cursor:pointer;

            box-shadow:
            0 10px 25px rgba(79,70,229,0.3);

            transition:0.3s;
        }

        button:hover{
            transform:translateY(-3px);
        }

        .result{

            margin-top:25px;
            padding:25px;

            text-align:center;

            background:#ecfdf5;

            border-radius:15px;

            box-shadow:
            0 10px 20px rgba(0,0,0,0.1);
        }

        .price{
            font-size:35px;
            color:#16a34a;
            font-weight:bold;
        }

        @media(max-width:700px){

            .grid{
                grid-template-columns:1fr;
            }

            .full{
                grid-column:auto;
            }
        }

    </style>

</head>

<body>

<div class="container">

    <div class="header">
        <h1>🏠 House Price Predictor</h1>
        <p>Linear Regression Model</p>
    </div>

    <div class="form-area">

        <form method="POST">

            <div class="grid">

                <div class="field">
                    <label>Square Footage</label>
                    <input type="number" name="Square_Footage" required>
                </div>

                <div class="field">
                    <label>Bedrooms</label>
                    <input type="number" name="Num_Bedrooms" required>
                </div>

                <div class="field">
                    <label>Bathrooms</label>
                    <input type="number" step="any" name="Num_Bathrooms" required>
                </div>

                <div class="field">
                    <label>Year Built</label>
                    <input type="number" name="Year_Built" required>
                </div>

                <div class="field">
                    <label>Lot Size</label>
                    <input type="number" step="any" name="Lot_Size" required>
                </div>

                <div class="field">
                    <label>Garage Size</label>
                    <input type="number" step="any" name="Garage_Size" required>
                </div>

                <div class="field full">

                    <label>Neighborhood Quality</label>

                    <select name="Neighborhood_Quality" required>

                        <option value="1">Poor</option>
                        <option value="2">Average</option>
                        <option value="3">Good</option>
                        <option value="4">Very Good</option>
                        <option value="5">Excellent</option>

                    </select>

                </div>

            </div>

            <button type="submit">
                Predict Price
            </button>

        </form>

        {% if prediction %}

        <div class="result">

            <h3>Predicted House Price</h3>

            <div class="price">
                ₹ {{ prediction }}
            </div>

        </div>

        {% endif %}

    </div>

</div>

</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None

    if request.method == "POST":

        try:

            data = np.array([[
                float(request.form["Square_Footage"]),
                float(request.form["Num_Bedrooms"]),
                float(request.form["Num_Bathrooms"]),
                float(request.form["Year_Built"]),
                float(request.form["Lot_Size"]),
                float(request.form["Garage_Size"]),
                float(request.form["Neighborhood_Quality"])
            ]])

            result = model.predict(data)[0]

            prediction = format(result, ",.2f")

        except Exception as e:
            prediction = "Error"

    return render_template_string(
        HTML,
        prediction=prediction
    )

app = app
