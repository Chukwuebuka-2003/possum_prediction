from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('possum.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    required_params = ['skullw', 'totlngth', 'footlngth', 'belly', 'chest', 'eye', 'age']
    if not all(param in data for param in required_params):
        return jsonify({'error': 'Missing required parameters'})

    # Perform prediction using the loaded model
    try:
        prediction = model.predict([[data['skullw'], data['totlngth'], data['footlngth'], data['belly'], data['chest'], data['eye'], data['age']]])
        # Return prediction as part of response
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
