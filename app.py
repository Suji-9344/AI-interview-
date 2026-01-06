from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import cv2
import base64
import numpy as np
from fer import FER
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load NLP model once
nlp_model = SentenceTransformer("all-MiniLM-L6-v2")

# Emotion detector
emotion_detector = FER(mtcnn=True)

# Sample correct answer (you can load from dataset)
CORRECT_ANSWER = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

def score_answer(user_answer, correct_answer):
    u = nlp_model.encode(user_answer, convert_to_tensor=True)
    c = nlp_model.encode(correct_answer, convert_to_tensor=True)
    return float(util.cos_sim(u, c))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # --------- VOICE TO TEXT ----------
    recognizer = sr.Recognizer()
    audio_file = request.files['audio']

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        user_text = recognizer.recognize_google(audio)

    # --------- FACE IMAGE ----------
    img_data = request.form['image']
    img_bytes = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --------- EMOTION & CONFIDENCE ----------
    emotion, emo_score = emotion_detector.top_emotion(img)

    confidence_map = {
        "happy": 0.9,
        "neutral": 0.7,
        "surprise": 0.8,
        "sad": 0.4,
        "fear": 0.3,
        "angry": 0.2,
        "disgust": 0.2
    }
    confidence_score = confidence_map.get(emotion, 0.5)

    # --------- NLP ANSWER SCORE ----------
    answer_score = score_answer(user_text, CORRECT_ANSWER)

    # --------- FINAL SCORE ----------
    final_score = (answer_score * 0.7) + (confidence_score * 0.3)

    return jsonify({
        "transcript": user_text,
        "emotion": emotion,
        "answer_score": round(answer_score, 2),
        "confidence_score": round(confidence_score, 2),
        "final_score": round(final_score, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
