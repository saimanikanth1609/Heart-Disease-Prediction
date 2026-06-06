from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("heart_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_input = np.array(input_features).reshape(1, -1)
    
    prediction = model.predict(final_input)

    if prediction[0] == 0:
        result = "The person is NOT having heart disease"
    else:
        result = "The person IS having heart disease"

    return render_template("result.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)