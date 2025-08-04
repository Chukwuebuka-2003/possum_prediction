# pyright: ignore[reportMissingImports]

from flask import Flask, render_template, request, jsonify # pyright: ignore[reportMissingImports]

import pickle

app = Flask(__name__)

# Load model
with open('eye_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET','POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            data = {
                'skullw': float(request.form['skullw']),
                'totlngth': float(request.form['totlngth']),
                'footlgth': float(request.form['footlgth']),
                'belly': float(request.form['belly']),
                'chest': float(request.form['chest']),
                'hdlngth': float(request.form['hdlngth']),
                'age': float(request.form['age']),
                'taill': float(request.form['taill'])  # Assuming 'tail' is also a feature
            }

            features = [[
    data['skullw'], data['totlngth'], data['footlgth'],
    data['belly'], data['chest'], data['hdlngth'], data['age'], data['taill']
           ]]
            prediction = round(model.predict(features)[0], 2)
        except Exception as e:
            prediction = f"Error: {e}"
          

    return render_template('index.html', prediction=prediction) # pyright: ignore[reportUndefinedVariable]

if __name__ == '__main__':
    app.run(debug=True)