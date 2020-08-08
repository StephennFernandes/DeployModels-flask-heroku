import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import send_from_directory
import math 

# Define a flask app
app = Flask(__name__)

dir_path = os.path.abspath(os.path.dirname(__file__)) #Get path
MODEL_PATH = dir_path + '/model'
model = tf.keras.models.load_model(MODEL_PATH) #load model

def sigmoid(value):
    return 1 / (1 + math. exp(-value))

def predict(img): #predict the output
    result = model.predict(img)
    result = sigmoid(result[0])
    print(result)
    if result < 0.5:
        return 0
    else:
        return 1



@app.route('/', methods=['GET'])
def index(): #home page
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload(): #upload image and predict the result
    if request.method == 'POST':
        image = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

        # Make prediction
        image1 = load_img(file_path, target_size=(160, 160))
        image1 = img_to_array(image1)
        image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))

        indices = {0: 'Cat', 1: 'Dog'}

        result = predict(image1)   
        label = indices[result]
        return render_template('predict.html', image_file_name = image.filename , label = label)
    return None 


@app.route('/upload/<filename>')
def send_file(filename):
    basepath = os.path.dirname(__file__)
    return send_from_directory(os.path.join(basepath, 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)

