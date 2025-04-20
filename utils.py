import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import yt_dlp
import glob


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    features = []

    try:
        features.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
    except:
        features.append(0.0)

    try:
        features.append(np.mean(librosa.feature.rms(y=y)))
    except:
        features.append(0.0)

    try:
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    except:
        features.append(0.0)

    try:
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    except:
        features.append(0.0)

    try:
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    except:
        features.append(0.0)

    try:
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    except:
        features.append(0.0)

    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features.append(np.mean(mfccs[i]))
    except:
        features.extend([0.0] * 13)

    while len(features) < 20:
        features.append(0.0)

    return np.array(features)


def train_model():
    X = []
    for _ in range(25):
        X.append(np.random.normal(loc=0.8, scale=0.05, size=20))  # mutlu
    for _ in range(25):
        X.append(np.random.normal(loc=0.2, scale=0.05, size=20))  # üzgün
    for _ in range(25):
        X.append(np.random.normal(loc=0.7, scale=0.1, size=20))   # enerjik
    for _ in range(25):
        X.append(np.random.normal(loc=0.3, scale=0.05, size=20))  # sakin

    y = (["mutlu"] * 25) + (["üzgün"] * 25) + (["enerjik"] * 25) + (["sakin"] * 25)
    clf = RandomForestClassifier()
    clf.fit(np.array(X), y)
    return clf


def download_youtube_audio(link, output_path='downloads'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Klasörü temizle (eski .wav dosyaları silinsin)
    for f in glob.glob(os.path.join(output_path, '*.wav')):
        os.remove(f)

    output_template = os.path.join(output_path, '%(title).30s.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

        # En yeni indirilen .wav dosyasını bul
        wav_files = sorted(glob.glob(os.path.join(output_path, '*.wav')), key=os.path.getctime, reverse=True)
        if wav_files:
            return wav_files[0]

        raise FileNotFoundError("WAV dosyası bulunamadı.")
    except Exception as e:
        raise RuntimeError(f"YouTube'dan ses indirme hatası: {e}")

