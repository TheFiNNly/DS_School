import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, url_for, request, redirect


model_I = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\model_reg_I.joblib")
model_OO = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\model_reg_OO.joblib")
standart_scaler_I = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\standart_scaler_I.joblib")
standart_scaler_OO = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\standart_scaler_OO.joblib")

KNN_district = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\KNN_district.joblib")
KNN_MO = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\KNN_MO.joblib")
standart_scaler_district = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\standart_scaler_district.joblib")
standart_scaler_MO = joblib.load("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\standart_scaler_MO.joblib")


data_price = pd.read_csv("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\data_mean.csv")
data_district = pd.read_csv("C:\\Users\\magvl\\блокноты\\final project\\notebook\\models\\data_area_mean.csv")
sub_area = pd.read_csv("C:\\Users\\magvl\\блокноты\\final project\\notebook\\data\\sub_area.csv")


def pred_price_def():
    global predict_price
    predict_price = []
    x = data_price['Value'].values

    if request.form['product_type_OwnerOccupier'] == 0:
        X_scaler_I = standart_scaler_I.transform(x.reshape(1, -1))
        pred_price = model_I.predict(X_scaler_I.reshape(1, -1))
        pred_price = np.e ** pred_price[0]
    else:
        X_scaler_OO = standart_scaler_OO.transform(x.reshape(1, -1))
        pred_price = model_OO.predict(X_scaler_OO.reshape(1, -1))
        pred_price = np.e ** pred_price[0]

    predict_price.append(round(pred_price))


def district_def():
    global predict_district, predict_MO
    predict_MO = []
    predict_district = []

    x = data_district['value'].values
    X_scaler_MO = standart_scaler_MO.transform(x.reshape(1, -1))
    X_scaler_district = standart_scaler_district.transform(x.reshape(1, -1))

    pred_MO = KNN_MO.predict(X_scaler_MO.reshape(1, -1))
    pred_district = KNN_district.predict(X_scaler_district.reshape(1, -1))

    pred_MO = sub_area['sub_area_rus'][sub_area['sub_area'] == pred_MO[0]].values

    predict_MO.append(pred_MO[0])
    predict_district.append(pred_district[0])


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home_page.html', title='Чек')


@app.route('/map')
def map_msk():
    return render_template('map.html', title='Средние цены на квартиры в Москве')


@app.route('/price', methods=['POST', 'GET'])
def price():
    return render_template('price.html')


@app.route('/predict_price', methods=['POST', 'GET'])
def predict_price():
    data_price['Value'][data_price['name'] == 'full_sq'] = request.form['full_sq']
    data_price['Value'][data_price['name'] == 'floor'] = request.form['floor']
    data_price['Value'][data_price['name'] == 'num_room'] = request.form['num_room']
    data_price['Value'][data_price['name'] == 'metro_min_avto'] = request.form['metro_min_avto']
    data_price['Value'][data_price['name'] == 'product_type_OwnerOccupier'] = request.form['product_type_OwnerOccupier']
    sub_area_value = sub_area['value'][sub_area['District'] == request.form['sub_area']].mean()
    data_price['Value'][data_price['name'] == 'sub_area'] = sub_area_value
    pred_price_def()
    return render_template('predict_price.html', predict=predict_price)


@app.route('/district')
def district():
    return render_template('district.html')


@app.route('/predict_district', methods=['POST', 'GET'])
def predict_district():
    data_district['value'][data_district['name'] == 'full_sq'] = request.form['full_sq']
    data_district['value'][data_district['name'] == 'num_room'] = request.form['num_room']
    data_district['value'][data_district['name'] == 'product_type_OwnerOccupier'] = request.form['product_type_OwnerOccupier']
    log_price = request.form['log_price_doc']
    data_district['value'][data_district['name'] == 'log_price_doc'] = np.log1p(int(log_price))
    district_def()
    return render_template('predict_district.html', predict_MO=predict_MO, predict_district=predict_district)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


