from flask import Flask, send_file, request,redirect,jsonify
import cv2
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from binascii import a2b_base64
import wave

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    
    emotion_analysis(custom[0])
    emotionMatrix = {"Angry":str(custom[0][0]),"Disgust":str(custom[0][1]),"Fear":str(custom[0][2]),"Happy":str(custom[0][3]),"Sad":str(custom[0][4]),"Surprise":str(custom[0][5]),"Neutral":str(custom[0][6])}
    response = {"matrix":emotionMatrix,"resultEmotion":result}
    return jsonify(response)
##    return send_file("./Plot.png")


def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    if(os.path.exists("Plot.png")):
        os.remove("Plot.png")
    plt.savefig("Plot.png")


@app.route('/captureimage',methods=['POST'])
@cross_origin()
def getiamge():
    a=request.get_data("image")
    a=a.decode("utf-8")
    a=a[a.find('base64,')+len('base64,'):a.find('------WebKitFormBoundary',a.find('base64,')+len('base64,'))]
    binary_data = a2b_base64(a)
    fd = open('image.png', 'wb')
    fd.write(binary_data)
    fd.close()
    clickCropSave()
    return redirect("/result")

@app.route('/captureaudio',methods=['POST'])
@cross_origin()
def getaudio():
    blob = request.files['file']
    name = "./output.wav"
    audio = wave.open(name, 'wb')
    audio.setnchannels(1)
    audio.setframerate(47000)
    audio.setsampwidth(4)
    audio.writeframes(blob.read())
    return redirect("/resultaudio")
    
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

def emotion_analysis_audio(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    bar_list = plt.bar(y_pos, emotions, align='center', alpha=0.5)
    bar_list[0].set_color('r')
    bar_list[1].set_color('pink')
    bar_list[2].set_color('orange')
    bar_list[3].set_color('green')
    bar_list[4].set_color('yellow')
    bar_list[5].set_color('black')
    if(os.path.exists("Plotaudio.png")):
        os.remove("Plotaudio.png")
    plt.savefig("Plotaudio.png")
    return send_file("Plotaudio.png")

@app.route('/resultaudio')
def resultaudio():
    model = load_model('Speech-Emotion-Recognition-Model.h5')
    data = {"mfcc":[]}
    y, sr = librosa.load('./output.wav')
    y = np.array(y)
    emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad')
    file = './output.wav'
    signal, sample_rate = librosa.load(file, sr)
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    data["mfcc"].append(np.asarray(mfcc))
    x = np.asarray(data['mfcc'])
    print(x)
    x = tf.keras.preprocessing.sequence.pad_sequences(x)
    print(x.shape)
    #     data["mfcc"].append(np.asarray(mfcc))
    custom = model.predict(x)
    result = [i * 100 for i in custom[0]]
    emotion = result.index(max(result))
    result = emotions[emotion]
    emotionMatrixAudio = {"Angry":str(custom[0][0]),"Disgust":str(custom[0][1]),"Fear":str(custom[0][2]),"Happy":str(custom[0][3]),"Neutral":str(custom[0][4]),"Sad":str(custom[0][5])}
    response = {"matrixAudio":emotionMatrixAudio,"resultEmotionAudio":result}
    return jsonify(response)
    #return emotion_analysis_audio(result[0])

@app.after_request # blueprint can also be app~~
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response
if __name__ == '__main__':
    app.run()
