from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the emotion classifier model
pipe_lr = joblib.load("C:\\Users\\Pavithra B\\Downloads\\emot\\emotion_classifier_pipe_lr.pkl")

# Emotion emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

# Function to predict emotions
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Emotion classification route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from form
        raw_text = request.form['text']
        
        # Predict emotion and probability
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
        confidence = np.max(probability)
        
        # Get the corresponding emoji
        emoji_icon = emotions_emoji_dict.get(prediction, "")
        
        # Render result on the webpage
        return render_template('result.html', 
                               raw_text=raw_text, 
                               prediction=prediction, 
                               emoji_icon=emoji_icon, 
                               confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
