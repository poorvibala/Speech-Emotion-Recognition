import os
import glob
import pickle
import numpy as np
import soundfile as sf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def extract_feature_file(file_path, mfcc=True, chroma=True, mel=True):
    try:
        with sf.SoundFile(file_path) as sound_file:
            X = sound_file.read(dtype="float32")
            sr = sound_file.samplerate
            if X is None or len(X) == 0:
                return None
            if X.ndim > 1:
                X = np.mean(X, axis=1)
            if len(X) < 2048:
                X = np.pad(X, (0, 2048 - len(X)), mode='constant')
            result = np.array([])
            if chroma:
                stft = np.abs(librosa.stft(X))
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
                result = np.hstack((result, chroma_feat))
            if mel:
                mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T, axis=0)
                result = np.hstack((result, mel_feat))
            return result
    except Exception:
        return None

data_dir = "dataset"
labels = []
features_list = []
for label_dir in sorted(os.listdir(data_dir)):
    full_dir = os.path.join(data_dir, label_dir)
    if not os.path.isdir(full_dir):
        continue
    wav_files = glob.glob(os.path.join(full_dir, "*.wav"))
    for wf in wav_files:
        feat = extract_feature_file(wf, mfcc=True, chroma=True, mel=True)
        if feat is None:
            continue
        features_list.append(feat)
        labels.append(label_dir)

X = np.array(features_list)
y = np.array(labels)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

with open("modelForPrediction.sav", "wb") as f:
    pickle.dump(clf, f)
with open("label_encoder.sav", "wb") as f:
    pickle.dump(le, f)

print("Saved modelForPrediction.sav and label_encoder.sav")
print("Training samples:", X_train.shape[0], "Feature length:", X_train.shape[1])
print("Test accuracy:", clf.score(X_test, y_test))
