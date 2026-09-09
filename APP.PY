from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear.pkl")
model = joblib.load(MODEL_PATH)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>House Price Predictor</title>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, Helvetica, sans-serif;
        }

        body {
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, #dbeafe, transparent 35%),
                radial-gradient(circle at bottom right, #ede9fe, transparent 35%),
                linear-gradient(135deg, #f8fafc, #eef2ff);

            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
        }

        .container {
            width: 100%;
            max-width: 950px;
            background: rgba(255, 255, 255, 0.92);
            border-radius: 28px;

            box-shadow:
                0 25px 60px rgba(15, 23, 42, 0.18),
                0 8px 20px rgba(99, 102, 241, 0.10);

            overflow: hidden;
            backdrop-filter: blur(12px);
        }

        .header {
            padding: 35px;
            text-align: center;

            background: linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            );

            color: white;
        }

        .header h1 {
            font-size: 34px;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }

        .header p {
            font-size: 15px;
            opacity: 0.9;
        }

        .form-area {
            padding: 38px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }

        .field {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: 700;
            color: #334155;
            margin-bottom: 9px;
            font-size: 14px;
        }

        input,
        select {
            width: 100%;
            padding: 14px 16px;

            border: 1px solid #dbe2ea;
            border-radius: 13px;

            background: #f8fafc;
            color: #1e293b;

            font-size: 15px;
            outline: none;

            transition: all 0.25s ease;

            box-shadow:
                inset 0 1px 2px rgba(0,0,0,0.03);
        }

        input:focus,
        select:focus {
            border-color: #6366f1;
            background: white;

            box-shadow:
                0 0 0 4px rgba(99,102,241,0.12),
                0 6px 18px rgba(99,102,241,0.08);

            transform: translateY(-1px);
        }

        select {
            cursor: pointer;
        }

        .full {
            grid-column: 1 / -1;
        }

        .predict-btn {
            width: 100%;
            margin-top: 30px;

            padding: 16px;

            border: none;
            border-radius: 14px;

            background: linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            );

            color: white;
            font-size: 17px;
            font-weight: 700;

            cursor: pointer;

            box-shadow:
                0 10px 25px rgba(79,70,229,0.30);

            transition: all 0.25s ease;
        }

        .predict-btn:hover {
            transform: translateY(-3px);

            box-shadow:
                0 15px 35px rgba(79,70,229,0.38);
        }

        .predict-btn:active {
            transform: translateY(0);
        }

        .result {
            margin-top: 28px;
            padding: 24px;

            border-radius: 18px;

            text-align: center;

            background: linear-gradient(
                135deg,
                #ecfdf5,
                #f0fdf4
            );

            border: 1px solid #bbf7d0;

            box-shadow:
                0 10px 25px rgba(34,197,94,0.10);
        }

        .result-title {
            color: #166534;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .price {
            color: #15803d;
            font-size: 32px;
            font-weight: 800;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            border-radius: 12px;

            background: #fef2f2;
            color: #b91c1c;

            text-align: center;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #64748b;
            font-size: 13px;
            background: #f8fafc;
        }

        @media (max-width: 700px) {
            body {
                padding: 15px;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            .full {
                grid-column: auto;
            }

            .header h1 {
                font-size: 27px;
            }

            .form-area {
                padding: 25px;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h1>🏠 House Price Predictor</h1>
        <p>AI-powered house price prediction using Linear Regression</p>
    </div>

    <div class="form-area">

        <form method="POST">

            <div class="grid">

                <div class="field">
                    <label>Square Footage</label>
                    <input
                        type="number"
                        name="Square_Footage"
                        placeholder="Example: 1800"
                        min="1"
                        step="any"
                        required
                    >
                </div>

                <div class="field">
                    <label>Number of Bedrooms</label>
                    <input
                        type="number"
                        name="Num_Bedrooms"
                        placeholder="Example: 3"
                        min="0"
                        step="1"
                        required
                    >
                </div>

                <div class="field">
                    <label>Number of Bathrooms</label>
                    <input
                        type="number"
                        name="Num_Bathrooms"
                        placeholder="Example: 2"
                        min="0"
                        step="any"
                        required
                    >
                </div>

                <div class="field">
                    <label>Year Built</label>
                    <input
                        type="number"
                        name="Year_Built"
                        placeholder="Example: 2015"
                        min="1800"
                        max="2100"
                        required
                    >
                </div>

                <div class="field">
                    <label>Lot Size</label>
                    <input
                        type="number"
                        name="Lot_Size"
                        placeholder="Example: 5000"
                        min="0"
                        step="any"
                        required
                    >
                </div>

                <div class="field">
                    <label>Garage Size</label>
                    <input
                        type="number"
                        name="Garage_Size"
                        placeholder="Example: 2"
                        min="0"
                        step="any"
                        required
                    >
                </div>

                <div class="field full">
                    <label>Neighborhood Quality</label>

                    <select name="Neighborhood_Quality" required>
                        <option value="">Select Neighborhood Quality</option>

                        <option value="1">Poor</option>
                        <option value="2">Average</option>
                        <option value="3">Good</option>
                        <option value="4">Very Good</option>
                        <option value="5">Excellent</option>
                    </select>
                </div>

            </div>

            <button class="predict-btn" type="submit">
                ✨ Predict House Price
            </button>

        </form>

        {% if prediction %}
        <div class="result">
            <div class="result-title">
                ESTIMATED HOUSE PRICE
            </div>

            <div class="price">
                ₹ {{ prediction }}
            </div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">
            {{ error }}
        </div>
        {% endif %}

    </div>

    <div class="footer">
        Linear Regression Model • Machine Learning Prediction
    </div>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            square_footage = float(request.form["Square_Footage"])
            bedrooms = float(request.form["Num_Bedrooms"])
            bathrooms = float(request.form["Num_Bathrooms"])
            year_built = float(request.form["Year_Built"])
            lot_size = float(request.form["Lot_Size"])
            garage_size = float(request.form["Garage_Size"])

            # Categorical value converted to numeric value
            neighborhood_quality = float(
                request.form["Neighborhood_Quality"]
            )

            # IMPORTANT:
            # Order must match model.feature_names_in_
            features = np.array([[
                square_footage,
                bedrooms,
                bathrooms,
                year_built,
                lot_size,
                garage_size,
                neighborhood_quality
            ]])

            result = model.predict(features)[0]

            prediction = f"{result:,.2f}"

        except Exception as e:
            error = "Please enter valid values. Prediction could not be generated."

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )


# Vercel uses this Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
