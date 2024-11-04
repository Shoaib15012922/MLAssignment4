from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle


app = Flask(__name__)
CORS(app)  


with open("naive_bayes_model.pkl", "rb") as nb_file:
    loaded_nb_model = pickle.load(nb_file)

with open("perceptron_model.pkl", "rb") as perc_file:
    loaded_perceptron_model = pickle.load(perc_file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model_type', 'naive_bayes')
    input_features = np.array([[data['age'], data['glucose'], data['insulin'], data['bmi']]])
    
    if model_type == 'naive_bayes':
        prediction = loaded_nb_model.predict(input_features)
    elif model_type == 'perceptron':
        prediction = loaded_perceptron_model.predict(input_features)
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    return jsonify({'diabetes_type': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)