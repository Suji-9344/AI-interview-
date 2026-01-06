import streamlit as st
import speech_recognition as sr
import numpy as np
from PIL import Image
import cv2
import tempfile
import time

st.title("🎤 Smart AI Interview Practice with Webcam")

# Sample question and correct answer
QUESTION = "Explain machine learning in simple terms."
CORRECT_ANSWER = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

# ---------------- HELPER FUNCTIONS ----------------
def score_answer(user_answer, correct_answer):
    """Simple scoring: ratio of common words"""
    user_words = set(user_answer.lower().split())
    correct_words = set(correct_answer.lower().split())
    common_words = user_words.intersection(correct_words)
    return round(len(common_words) / len(correct_words), 2)

def confidence_from_speech(text):
    length_score = min(len(text.split()) / 25, 1.0)
    return round(length_score, 2)

def confidence_from_face(img_array):
    # Simple rule: if face is detected, confident
    if img_array is not None:
        return 0.8
    else:
        return 0.4

# ---------------- STREAMLIT INTERFACE ----------------
st.subheader("Question for You:")
st.write(QUESTION)

# 1️⃣ Webcam capture
st.subheader("Capture Your Face")
picture = st.camera_input("Take a picture")

# 2️⃣ Microphone record
st.subheader("Record Your Answer")
audio_file = st.audio_input("Click to record your answer (max 10 sec)", format="wav")

# 3️⃣ Evaluate
if st.button("Evaluate Answer"):

    if picture is None or audio_file is None:
        st.error("Please capture your face and record your answer.")
    else:
        # ---------- PROCESS AUDIO ----------
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

        # ---------- PROCESS IMAGE ----------
        img = Image.open(picture).convert("RGB")
        st.image(img, caption="Your Captured Face", use_column_width=True)

        # Convert PIL to array for face confidence
        img_array = np.array(img)

        # ---------- SCORING ----------
        answer_score = score_answer(user_text, CORRECT_ANSWER)
        speech_conf = confidence_from_speech(user_text)
        face_conf = confidence_from_face(img_array)
        confidence_score = round((speech_conf + face_conf)/2, 2)
        final_score = round((answer_score*0.7) + (confidence_score*0.3), 2)

        # ---------- RESULTS ----------
        st.subheader("📊 Interview Results")
        st.write("Answer Score:", answer_score)
        st.write("Confidence Score:", confidence_score)
        st.write("⭐ Final Score:", final_score)
