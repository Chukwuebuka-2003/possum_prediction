from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model

with open('belly_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            data = {
                'hdlngth': float(request.form['hdlngth']),
                'skullw': float(request.form['skullw']),
                'totlngth': float(request.form['totlngth']),
                'taill': float(request.form['taill']),
                'footlgth': float(request.form['footlgth']),
                'chest': float(request.form['chest']),
                'eye': float(request.form['eye']),
                'age': float(request.form['age']),
            }

            features = [[
                data['hdlngth'], data['skullw'], data['totlngth'],
                data['taill'], data['footlgth'], data['chest'],
                data['eye'], data['age']
            ]]
            prediction = round(model.predict(features)[0], 2)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('belly_form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)