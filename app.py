import streamlit as st
import speech_recognition as sr
import numpy as np
from PIL import Image
import io
import base64

st.title("🎤 Smart AI Interview Practice")

# Sample correct answer
CORRECT_ANSWER = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

# ---------------- HELPER FUNCTIONS ----------------
def score_answer(user_answer, correct_answer):
    """Simple scoring function without transformers"""
    user_words = set(user_answer.lower().split())
    correct_words = set(correct_answer.lower().split())
    common_words = user_words.intersection(correct_words)
    return round(len(common_words) / len(correct_words), 2)

def confidence_from_speech(text):
    """Confidence based on length of answer"""
    length_score = min(len(text.split()) / 25, 1.0)
    return round(length_score, 2)

def confidence_from_face(image):
    """Confidence based on face presence (simple)"""
    if image is not None:
        return 0.8
    else:
        return 0.4

# ---------------- USER INPUT ----------------
st.subheader("Record Your Answer")

uploaded_audio = st.file_uploader("Upload your recorded audio (.wav)", type=["wav"])
uploaded_image = st.file_uploader("Upload your webcam image (.jpg/.png)", type=["jpg","png"])

if st.button("Evaluate Answer"):

    if uploaded_audio is None or uploaded_image is None:
        st.error("Please upload both audio and image")
    else:
        # ---------- AUDIO ----------
        recognizer = sr.Recognizer()
        with sr.AudioFile(uploaded_audio) as source:
            audio = recognizer.record(source)
            try:
                user_text = recognizer.recognize_google(audio)
            except:
                user_text = ""

        st.write("🗣️ Your Answer:", user_text)

        # ---------- IMAGE ----------
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Webcam Image", use_column_width=True)

        # ---------- SCORING ----------
        answer_score = score_answer(user_text, CORRECT_ANSWER)
        speech_conf = confidence_from_speech(user_text)
        face_conf = confidence_from_face(img)
        confidence_score = round((speech_conf + face_conf)/2, 2)
        final_score = round((answer_score*0.7) + (confidence_score*0.3), 2)

        # ---------- OUTPUT ----------
        st.subheader("📊 Results")
        st.write("Answer Score:", answer_score)
        st.write("Confidence Score:", confidence_score)
        st.write("⭐ Final Score:", final_score)
