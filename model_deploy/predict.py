import pickle

from flask import Flask
from flask import request
from flask import jsonify


app = Flask('churn')

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as file_in:
    dv, model = pickle.load(file_in)


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    churn = bool(y_pred >= 0.5)

    result = {
        'churn_probabilty': float(y_pred),
        'churn': churn
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
