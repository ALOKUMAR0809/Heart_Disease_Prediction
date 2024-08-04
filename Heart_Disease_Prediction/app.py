# app.py

from flask import Flask, render_template, request, jsonify
import numpy as np
from collections import Counter
import pickle

app = Flask(__name__)

with open('model_dt.pkl', 'rb') as f_dt:
        model_dt = pickle.load(f_dt)

with open('model_rf.pkl', 'rb') as f_rf:
        model_rf = pickle.load(f_rf)

with open('model_knn.pkl', 'rb') as f_knn:
        model_knn = pickle.load(f_knn)

with open('model_nb.pkl', 'rb') as f_nb:
        model_nb = pickle.load(f_nb)

with open('model_lg.pkl', 'rb') as f_lg:
        model_lg = pickle.load(f_lg)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the feature values from the request
    features = [float(x) for x in request.form.values()]
    # Preprocess the features as needed (e.g., scaling)
    features = np.array(features).reshape(1, -1)

    # Make predictions using the models
    prediction_dt = model_dt.predict(features)[0]
    prediction_rf = model_rf.predict(features)[0]
    knn_predictions = model_knn.predict(features)
    prediction_nb = model_nb.predict(features)[0]
    prediction_lg = model_lg.predict(features)[0]

    # Majority voting
    predictions = [prediction_dt, prediction_rf, knn_predictions[0], prediction_nb, prediction_lg]
    majority_vote = Counter(predictions).most_common(1)[0][0]

    # Return the predictions in JSON format
    return jsonify({
        'decision_tree': int(prediction_dt),
        'random_forest': int(prediction_rf),
        'knn': int(knn_predictions[0]),
        'naive_bayes': int(prediction_nb),
        'logistic_regression': int(prediction_lg),
        'majority_vote': int(majority_vote)
    })

if __name__ == '__main__':
    app.run(debug=True)
