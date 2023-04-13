import tensorflow as tf
from typing import List, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from flask import Flask, request, jsonify
from models.simple_models import simple_heuristic

app = Flask(__name__)


def load_models() -> Tuple[DecisionTreeClassifier, LogisticRegression, MinMaxScaler, tf.keras.models.Model]:
    """
    This function loads the trained models from disk.

    Returns:
        Tuple[DecisionTreeClassifier, LogisticRegression, MinMaxScaler, tf.keras.models.Model]: The trained models.
    """
    #Load models
    decision_tree = joblib.load("./models/checkpoints/decision_tree.joblib")
    logistic_regression = joblib.load("./models/checkpoints/logistic_regression.joblib")
    regression_scaler = joblib.load("./models/checkpoints/regression_scaler.joblib")
    neural_net = tf.keras.models.load_model("./models/checkpoints/neural_net.h5")

    return decision_tree, logistic_regression, regression_scaler, neural_net


def validate_input(input_data: List[int]) -> bool:
    """
    This function validates the input data.

    Args:
        input_data (List[int]): The input data to validate.

    Returns:
        bool: True if the input data is valid, False otherwise.
    """
    if len(input_data) != 54:
        return False
    for i in range(11):
        if not isinstance(input_data[i], (int, float)):
            return False
    for i in range(11, 54):
        if not isinstance(input_data[i], (int, float)):
            return False
        if input_data[i] not in [0, 1]:
            print(f"{input_data[i]} is not a 0 or 1")
            return False
    return True


@app.route('/predict', methods=['POST'])
def predict() -> int:
    """
    This function predicts the target of a data point.

    Args:
        name (str): The name of the model to use.
        data (List): The data to predict on.

    Returns:
        int: The predicted target.
    """
    #Load models
    decision_tree, logistic_regression, regression_scaler, neural_net = load_models()

    # Validate the input data
    if not request.json or not validate_input(request.json['features']):
        return jsonify({'error': 'Invalid input data'}), 400

    # Get the selected model from the request data
    selected_model = request.json['model']

    # Get the input features from the request data
    input_features = np.array(request.json['features'])

    # Call the appropriate machine learning model based on the user's choice
    if selected_model == 'heuristic':
        prediction = simple_heuristic(input_features[0])
    elif selected_model == 'tree':
        prediction = int(decision_tree.predict(input_features.reshape(1, -1)))
    elif selected_model == 'logistic_regression':
        prediction = int(logistic_regression.predict(regression_scaler.transform(input_features.reshape(1, -1))))
    elif selected_model == 'neural_network':
        prediction = int((neural_net.predict(input_features.reshape(1, -1)).argmax(axis=-1) + 1)[0])
    else:
        return jsonify({'error': 'Invalid model choice'}), 400

    # Return the prediction
    return jsonify({'prediction': prediction}), 200
    

if __name__ == '__main__':
    #Load models
    app.run()