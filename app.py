import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import send_from_directory
import os
import tensorflow
import numpy as np
from flask import request
import sys
from flask import Flask, render_template, url_for, flash, redirect
import joblib


app = Flask(__name__, template_folder='template')


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'


model = load_model('model111.h5')
xrayModel = load_model("x_model.h5")


def api1(full_path):
    data = image.load_img(full_path, target_size=(150, 150, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    predicted = xrayModel.predict(data)
    return predicted


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected',
                       2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Malaria"))


@app.route('/upload11', methods=['POST', 'GET'])
def upload11_file():

    if request.method == 'GET':
        return render_template('index2.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            result = result[0][0]
            if(result > 0.5):
                label = indices[1]
                accuracy = result
            else:
                label = indices[0]
                accuracy = (1-result)*100
            return render_template('predict1.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Pneumonia"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/diabetes")
def diabetes():
    # if form.validate_on_submit():
    return render_template("diabetes.html")


@app.route("/Pneumonia")
def Pneumonia():
    return render_template("index2.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if(size == 8):  # Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif(size == 30):  # Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size == 12):  # Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif(size == 10):
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif(size == 11):  # Heart
        loaded_model = joblib.load("model2")
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list) == 30):  # Cancer
            result = ValuePredictor(to_predict_list, 30)
        elif(len(to_predict_list) == 8):  # Daiabtes
            result = ValuePredictor(to_predict_list, 8)
        elif(len(to_predict_list) == 12):
            result = ValuePredictor(to_predict_list, 12)
        elif(len(to_predict_list) == 11):
            result = ValuePredictor(to_predict_list, 11)
            # if int(result)==1:
            #   prediction ='diabetes'
            # else:
            #   prediction='Healthy'
        elif(len(to_predict_list) == 10):
            result = ValuePredictor(to_predict_list, 10)
    if(int(result) == 1):
        prediction = 'Sorry ! Suffering'
    else:
        prediction = 'Congrats ! you are Healthy'
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
