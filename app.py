import streamlit as st
import speech_recognition as sr
from PIL import Image
import numpy as np
import tempfile

st.title("🎤 AI Interview Practice (Camera + Voice)")

# Question and correct answer
QUESTION = "Explain machine learning in simple terms."
CORRECT_ANSWER = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

st.subheader("Question for You:")
st.write(QUESTION)

# 1️⃣ Capture webcam image
st.subheader("Capture Your Face")
picture = st.camera_input("Take a picture")

# 2️⃣ Upload audio file
st.subheader("Record Your Answer")
audio_file = st.file_uploader("Upload your answer (.wav)", type=["wav"])

# ---------------- SCORING ----------------
def score_answer(user_answer, correct_answer):
    user_words = set(user_answer.lower().split())
    correct_words = set(correct_answer.lower().split())
    common_words = user_words.intersection(correct_words)
    return round(len(common_words)/len(correct_words), 2)

def confidence_from_speech(text):
    return round(min(len(text.split())/25, 1.0), 2)

def confidence_from_face(img_array):
    return 0.8 if img_array is not None else 0.4

# 3️⃣ Evaluate
if st.button("Evaluate Answer"):
    if picture is None or audio_file is None:
        st.error("Please capture your face and upload your answer audio")
    else:
        # Process audio
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_file.read())
            tmp_audio_path = tmp_audio.name

        with sr.AudioFile(tmp_audio_path) as source:
            audio = recognizer.record(source)
            try:
                user_text = recognizer.recognize_google(audio)
            except:
                user_text = ""

        st.write("🗣️ Your Answer:", user_text)

        # Process image
        img = Image.open(picture).convert("RGB")
        st.image(img, caption="Captured Face", use_column_width=True)
        img_array = np.array(img)

        # Scoring
        answer_score = score_answer(user_text, CORRECT_ANSWER)
        speech_conf = confidence_from_speech(user_text)
        face_conf = confidence_from_face(img_array)
        confidence_score = round((speech_conf + face_conf)/2, 2)
        final_score = round((answer_score*0.7) + (confidence_score*0.3), 2)

        # Results
        st.subheader("📊 Interview Results")
        st.write("Answer Score:", answer_score)
        st.write("Confidence Score:", confidence_score)
        st.write("⭐ Final Score:", final_score)
