# Final working app.py for Speech Emotion Recognition using MLP and Streamlit

import os
import librosa
import soundfile as sf
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# ---------------------------
# Set root and define paths
# ---------------------------
ROOT = "F:/Speech emotion recognition/dataset"

# ---------------------------
# Emotion Mapping
# ---------------------------
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# ---------------------------
# Feature Extraction Function
# ---------------------------
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        with sf.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            if len(X) < 2048:
                print(f"⚠️ Skipping too short file: {file_name}")
                return None

            result = np.array([])
            if chroma:
                stft = np.abs(librosa.stft(X))

            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))

            if chroma:
                chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma_feat))

            if mel:
                mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel_feat))

            return result
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        return None

# ---------------------------
# Dataset Loader
# ---------------------------
def load_dataset(test_size=0.2):
    x, y = [], []
    files = glob.glob(ROOT + "/Actor_*/*.wav")
    print(f"🎧 Found {len(files)} audio files")

    for file in files:
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]
        emotion = emotions.get(emotion_code)

        if emotion not in observed_emotions:
            continue

        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        if features is not None:
            x.append(features)
            y.append(emotion)

    if len(x) == 0:
        raise ValueError("❌ No valid features extracted. Please check your audio path or file format.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return train_test_split(np.array(x), np.array(y_encoded), test_size=test_size, random_state=9), label_encoder

# ---------------------------
# Train and Save Model
# ---------------------------
(X_train, X_test, y_train_encoded, y_test_encoded), label_encoder = load_dataset()

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(X_train, y_train_encoded)

# Save model and encoder
with open('modelForPrediction.sav', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.sav', 'wb') as f:
    pickle.dump(label_encoder, f)

# ---------------------------
# Streamlit App Starts Here
# ---------------------------
@st.cache_resource

def load_model():
    model = pickle.load(open("modelForPrediction.sav", "rb"))
    label_encoder = pickle.load(open("label_encoder.sav", "rb"))
    return model, label_encoder

model, label_encoder = load_model()

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎤 Speech Emotion Recognition")
st.markdown("Upload a `.wav` file to predict the emotion using the trained MLP model.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format='audio/wav')

    features = extract_feature(temp_path, mfcc=True, chroma=True, mel=True)
    if features is None:
        st.error("❌ Failed to extract features from the uploaded audio.")
    else:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎯 Predicted Emotion: **{predicted_emotion.upper()}**")
        emotion_emoji = {
            "calm": "😌", "happy": "😄", "fearful": "😨", "disgust": "🤢"
        }
        if predicted_emotion in emotion_emoji:
            st.markdown(f"### {emotion_emoji[predicted_emotion]}")
