from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import base64
from PIL import Image
import io

app = Flask(__name__)

# Sample correct answer
CORRECT_ANSWER = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

# ---------------- HELPER FUNCTIONS ----------------
def score_answer(user_answer, correct_answer):
    """
    Simple scoring function without transformers:
    Measures ratio of correct words present
    """
    user_words = set(user_answer.lower().split())
    correct_words = set(correct_answer.lower().split())
    common_words = user_words.intersection(correct_words)
    return round(len(common_words) / len(correct_words), 2)

def confidence_from_speech(text):
    """
    Confidence based on length of answer (simple proxy)
    """
    length_score = min(len(text.split()) / 25, 1.0)
    return round(length_score, 2)

def confidence_from_face(image):
    """
    Confidence based on face presence (very simple)
    """
    if image is not None:
        return 0.8
    else:
        return 0.4

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # ---------- AUDIO ----------
    audio_file = request.files['audio']
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            user_text = recognizer.recognize_google(audio)
        except:
            user_text = ""  # fallback

    # ---------- IMAGE ----------
    img_data = request.form['image']
    img_bytes = base64.b64decode(img_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # ---------- SCORING ----------
    answer_score = score_answer(user_text, CORRECT_ANSWER)
    speech_conf = confidence_from_speech(user_text)
    face_conf = confidence_from_face(img)
    confidence_score = round((speech_conf + face_conf) / 2, 2)
    final_score = round((answer_score * 0.7) + (confidence_score * 0.3), 2)

    # ---------- RETURN RESULTS ----------
    return jsonify({
        "transcript": user_text,
        "answer_score": answer_score,
        "confidence_score": confidence_score,
        "final_score": final_score
    })

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
