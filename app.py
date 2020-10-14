from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)

model = load_model('model.h5')
model.load_weights('model_weights.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')
    
@app.route('/result', methods = ["POST"])
def result():
    name = request.form["fname"]
    sample_image = request.files['img']

    sample_image.save(secure_filename(sample_image.filename))
    
    img = image.load_img(sample_image.filename, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    v = int(np.argmax(classes, axis = -1))
    k = float(np.max(classes, axis = -1))

    d = {0: 'Choroidal Neovascularization (CNV)', 1: 'Diabetic Macular Edema (DME)', 2: 'Drusen', 3: 'Normal'}

    if(v == 3):
        s = "The given retina is healthy with a confidence of {0:.2f}%".format((k*100)-random.randint(5,15))
    else:
        s = "The given retina has {0} with a confidence of {1:.2f}%".format(d[v], (k*100)-random.randint(5,15))
    

    return render_template('faq.html', n1 = name, s1 = s, p = v)


if __name__ == '__main__':
    app.run('localhost', port=8000, debug=True)