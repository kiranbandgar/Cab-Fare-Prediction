import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__,template_folder='Template')

model = joblib.load(open('Cab fare prediction.pkl', 'rb'))


@app.route('/',methods=['GET'])
def home():
    render_template('index.html')


@app.route('/prediction',methods=['POST'])
def prediction():
    if request.method == 'POST':
        passenger = int(request.form['Passenger'])
        distance = float(request.form['Distance'])
        year = int(request.form['Year'])
        month = int(request.form['Month'])
        hour = int(request.form['Hour'])

        arr = np.array([[passenger, np.log(distance)+1, year, month, hour]])
        prediction = model.predict(arr)

    return render_template('after.html', data=prediction)


if __name__ == '__main__':
    app.run()


