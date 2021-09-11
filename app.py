
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    int_features =request.form['Title']
    final_features = [int_features]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Source: {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
