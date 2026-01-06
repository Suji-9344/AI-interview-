from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import base64
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io

app = Flask(__name__)

# Load NLP model
nlp_model = SentenceTransformer("all-MiniLM-L6-v2")

CORRECT_ANSWER = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

def score_answer(user_answer, correct_answer):
    u = nlp_model.encode(user_answer, convert_to_tensor=True)
    c = nlp_model.encode(correct_answer, convert_to_tensor=True)
    return float(util.cos_sim(u, c))

def confidence_from_speech(text):
    length_score = min(len(text.split()) / 25, 1.0)
    return round(length_score, 2)

def confidence_from_face(image):
    # Simple rule-based confidence (face detected → confident)
    return 0.8 if image is not None else 0.4

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # -------- AUDIO --------
    recognizer = sr.Recognizer()
    audio_file = request.files['audio']

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        user_text = recognizer.recognize_google(audio)

    # -------- IMAGE --------
    img_data = request.form['image']
    img_bytes = base64.b64decode(img_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # -------- SCORES --------
    answer_score = score_answer(user_text, CORRECT_ANSWER)
    speech_conf = confidence_from_speech(user_text)
    face_conf = confidence_from_face(img)

    confidence_score = round((speech_conf + face_conf) / 2, 2)
    final_score = round((answer_score * 0.7) + (confidence_score * 0.3), 2)

    return jsonify({
        "transcript": user_text,
        "answer_score": round(answer_score, 2),
        "confidence_score": confidence_score,
        "final_score": final_score
    })

if __name__ == "__main__":
    app.run(debug=True)
