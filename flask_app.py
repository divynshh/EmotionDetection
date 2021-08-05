from flask import Flask, send_file, request,redirect,jsonify
import cv2
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)

@app.route('/result')
def hello_world():
    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    model = load_model('model25.h5')

    file = "./face.jpg"
    true_image = image.load_img(file)
    img = image.load_img(file, grayscale=True, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = model.predict(x)
    result = [i * 100 for i in custom[0]]
    emotion = result.index(max(result))
    result = emotions[emotion]
    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);
    plt.gray()
    emotionMatrix
    emotion_analysis(custom[0])
    response = {"matrix":custom[0].tolist(),"resultEmotion":result}
    return jsonify(response)
    #return send_file("./Plot.png")


def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.savefig("Plot.png")


@app.route('/captureimage',methods=['POST'])
@cross_origin()
def getiamge():
    print("in request")
    file = request.files['image']
    print("dikkat")
    file.save("image.png")
    clickCropSave()
    return redirect("/result")
    
def clickCropSave():
    imagepath="image.png"
    img = cv2.imread(imagepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    faces = img[y:y + h, x:x + w]
    cv2.imwrite('./face.jpg', faces)

@app.after_request # blueprint can also be app~~
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response
if __name__ == '__main__':
    app.run()
