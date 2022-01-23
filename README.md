Emotion Detector Backend is based on selection machine learning models, hosted using flask microframework. Emotion Detector UI which is based on Angular JS sends the image and audio to this flask endpoints 
# Prerequisites
This flask based application requires Python 3.x
## Flask App Installation
### Install dependencies using requirements.txt
```bash
pip install -r requirements.txt
```
## Get the Code
```
git clone 'https://github.optum.com/dchauh24/EmotionDetection-Backend'
```
## Run the flask app to enable web service. 
```
cd EmotionDetection
python flask_app.py
```
## After succesfully running the flask server
We can see the IP address of our flask app as
```
* Running on <IP Address with port number> (Press CTRL+C to quit)
```
Copy this IP address and port number and replace with the ones on EmotionDetectorUI's app.component.ts on line numbers 145,162
