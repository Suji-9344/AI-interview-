import streamlit as st
import speech_recognition as sr
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="🎤 AI Interview Practice", layout="wide")
st.title("🎤 AI Interview Practice (Video + Voice)")

# ---------------- QUESTION ----------------
QUESTION = "Explain machine learning in simple terms."
CORRECT_ANSWER = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

st.subheader("Question for You:")
st.write(QUESTION)

# ---------------- HELPER FUNCTIONS ----------------
def score_answer(user_answer, correct_answer):
    user_words = set(user_answer.lower().split())
    correct_words = set(correct_answer.lower().split())
    common_words = user_words.intersection(correct_words)
    return round(len(common_words)/len(correct_words),2)

def confidence_from_speech(text):
    length_score = min(len(text.split())/25,1.0)
    return round(length_score,2)

def confidence_from_face(img_array):
    return 0.8 if img_array is not None else 0.4

# ---------------- VIDEO CAPTURE ----------------
st.subheader("Record Your Video Answer (Click Start)")

webrtc_ctx = webrtc_streamer(
    key="video",
    mode=WebRtcMode.SENDRECV,   # allows video + audio
    media_stream_constraints={"video": True, "audio": True},
)

# ---------------- PROCESS VIDEO ----------------
if st.button("Evaluate Answer"):
    if webrtc_ctx.state.playing:
        st.info("Please stop recording first by clicking 'Stop' in the video window.")
    elif not webrtc_ctx.video_receiver:
        st.error("No video recorded. Please record your answer first.")
    else:
        # Process audio frames
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if audio_frames:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                for frame in audio_frames:
                    tmp_audio_file.write(frame.to_ndarray().tobytes())
                tmp_audio_path = tmp_audio_file.name

            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp_audio_path) as source:
                audio = recognizer.record(source)
                try:
                    user_text = recognizer.recognize_google(audio)
                except:
                    user_text = ""
        else:
            user_text = ""

        st.write("🗣️ Your Answer:", user_text)

        # Process first video frame for confidence
        video_frame = webrtc_ctx.video_receiver.get_frame()
        if video_frame is not None:
            img_array = video_frame.to_ndarray(format="rgb24")
            img = Image.fromarray(img_array)
            st.image(img, caption="Captured Frame", use_column_width=True)
        else:
            img_array = None

        # ---------- SCORING ----------
        answer_score = score_answer(user_text, CORRECT_ANSWER)
        speech_conf = confidence_from_speech(user_text)
        face_conf = confidence_from_face(img_array)
        confidence_score = round((speech_conf + face_conf)/2,2)
        final_score = round((answer_score*0.7) + (confidence_score*0.3),2)

        # ---------- RESULTS ----------
        st.subheader("📊 Interview Results")
        st.write("Answer Score:", answer_score)
        st.write("Confidence Score:", confidence_score)
        st.write("⭐ Final Score:", final_score)
