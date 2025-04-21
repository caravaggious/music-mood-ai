import streamlit as st
import numpy as np
import os
from utils import extract_features, train_model, download_youtube_audio
import joblib
import matplotlib.pyplot as plt

# Sayfa yapılandırması
st.set_page_config(page_title="Müzik Ruh Hali Tahmini", layout="centered")

# Stil tanımı (sade ve modern Google tarzı)
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    h1, h2, p {
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    div.stButton > button:first-child {
        background-color: #4285F4;
        color: white;
        padding: 0.6em 1.5em;
        font-size: 16px;
        font-weight: 500;
        border: none;
        border-radius: 6px;
        margin-top: 10px;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #357AE8;
    }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("""
    <h1 style='text-align: center;'>🎵 Ruh Hali Tahmini</h1>
    <p style='text-align: center;'>YouTube videosunun sesinden müziğin ruh halini tahmin edin.</p>
""", unsafe_allow_html=True)

# 🎨 Pasta grafik fonksiyonu
def show_pie_chart(probs, labels, prediction):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#34A853', '#EA4335', '#FBBC05', '#4285F4']
    explode = [0.07 if p == max(probs) else 0 for p in probs]

    wedges, texts, autotexts = ax.pie(
        probs,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        textprops=dict(color="black", fontsize=12),
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    plt.setp(autotexts, size=13, weight="bold")
    ax.set_title(f"🎧 Tahmin: {prediction}", fontsize=16, weight="bold")
    ax.axis('equal')
    st.pyplot(fig)

# 📺 YouTube linkiyle analiz
st.markdown("---")
st.subheader("📺 YouTube Linki ile Analiz")

with st.form(key="youtube_form"):
    youtube_link = st.text_input("YouTube video linkini girin:")
    submit_button = st.form_submit_button(label="Tahmin Et")

if submit_button and youtube_link:
    try:
        with st.spinner("🎶 Video işleniyor..."):
            wav_path = download_youtube_audio(youtube_link, "yt_audio.wav")
            features = extract_features(wav_path)

            if os.path.exists("mood_model.pkl"):
                model = joblib.load("mood_model.pkl")
                prediction = model.predict([features])[0]
                probs = model.predict_proba([features])[0]
                labels = model.classes_

                st.success(f"🎧 Tahmin edilen ruh hali: **{prediction}**")
                show_pie_chart(probs, labels, prediction)
            else:
                st.error("❌ Model bulunamadı. Lütfen modeli eğitin.")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

# ℹ️ Hakkında bölümü
with st.expander("ℹ️ Hakkında", expanded=False):
    st.markdown("""
**Bu Uygulama Hakkında**  
YouTube üzerinden paylaştığınız müziklerin ses özelliklerini analiz ederek hangi ruh haliyle uyumlu olduğunu tahmin eder. Random Forest algoritması ile eğitilmiş basit bir model kullanır.

**Kategoriler:**
- Mutlu
- Üzgün
- Enerjik
- Sakin

**Teknolojiler:**
Python • Streamlit • librosa • yt-dlp • scikit-learn
    """)
