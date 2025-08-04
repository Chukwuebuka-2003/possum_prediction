from flask import Flask, render_template, request, jsonify 

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
                'hdlngth': float(request.form['hdlngth']),
                'skullw': float(request.form['skullw']),
                'totlngth': float(request.form['totlngth']),
                'taill': float(request.form['taill']),
                'footlgth': float(request.form['footlgth']),
                'belly': float(request.form['belly']),
                'eye': float(request.form['chest']),  # Changed from 'eye' to 'chest'
                'age': float(request.form['age']),
            }

            features = [[
                data['hdlngth'], data['skullw'], data['totlngth'],
                data['taill'], data['footlgth'], data['belly'],
                data['chest'], data['age']
            ]]
            prediction = round(model.predict(features)[0], 2)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction) 

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    
    
    
    