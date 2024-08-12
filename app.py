from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('./model/linear_model.pkl', 'rb'))

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    avg_area_income = float(request.form.get('avg_area_income'))
    avg_area_house_age = float(request.form.get('avg_area_house_age'))
    avg_area_num_rooms = float(request.form.get('avg_area_num_rooms'))
    avg_area_num_bedrooms = float(request.form.get('avg_area_num_bedrooms'))
    area_population = float(request.form.get('area_population'))
    house_features = [avg_area_income, avg_area_house_age, avg_area_num_rooms, avg_area_num_bedrooms, area_population]
    features = [np.array(house_features)]
    prediction = model.predict(features)

    return render_template('index.html', price_text = 'Price: ${:,.2f}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)