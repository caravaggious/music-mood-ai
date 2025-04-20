import streamlit as st
import numpy as np
import os
from utils import extract_features, train_model, download_youtube_audio
import joblib
import matplotlib.pyplot as plt

# Sayfa yapılandırması
st.set_page_config(page_title="Müzik Ruh Hali Tahmini", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🎵 Müzik Ruh Hali Tahmin Sistemi</h1>
    <p style='text-align: center; font-size:18px;'>YouTube linki ile müziğin ruh halini tahmin edin.</p>
""", unsafe_allow_html=True)

# 🎨 Pasta grafik fonksiyonu
def show_pie_chart(probs, labels, prediction):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#FFD700', '#FF6F61', '#87CEFA', '#90EE90']
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
    ax.set_title(f"🎧 Ruh Hali: {prediction}", fontsize=16, weight="bold")
    ax.axis('equal')
    st.pyplot(fig)

# 📺 YouTube linkiyle analiz
st.markdown("---")
st.markdown("<h2 style='color:#4B8BBE;'>📺 YouTube Linkiyle Ruh Hali Analizi</h2>", unsafe_allow_html=True)

with st.form(key="youtube_form"):
    youtube_link = st.text_input("🎬 Lütfen analiz etmek istediğiniz YouTube video linkini girin:")
    submit_button = st.form_submit_button(label="🎯 Tahmini Başlat")

if submit_button and youtube_link:
    try:
        wav_path = download_youtube_audio(youtube_link, "yt_audio.wav")
        features = extract_features(wav_path)

        if os.path.exists("mood_model.pkl"):
            model = joblib.load("mood_model.pkl")
            prediction = model.predict([features])[0]
            probs = model.predict_proba([features])[0]
            labels = model.classes_

            st.success(f"🎧 Tahmin edilen ruh hali: **{prediction}**")
            show_pie_chart(probs, labels, prediction)

            if max(probs) < 0.6:
                st.warning("⚠️ Bu tahminin güvenirliği düşük olabilir. Farklı bir şarkı deneyin.")
        else:
            st.error("❌ Önce modeli eğitmelisiniz!")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

# 🎯 Model eğitme özelliği (isteğe bağlı)
# if st.button("Modeli Oluştur / Güncelle"):
#     model = train_model()
#     joblib.dump(model, "mood_model.pkl")
#     st.success("✅ Model başarıyla eğitildi ve kaydedildi.")

# 📥 Manuel müzik dosyası yükleme (isteğe bağlı)
# uploaded_file = st.file_uploader("Bir müzik dosyası yükle (.wav)", type=["wav"])
# if uploaded_file is not None:
#     file_path = "uploaded_audio.wav"
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.read())

#     features = extract_features(file_path)

#     if os.path.exists("mood_model.pkl"):
#         model = joblib.load("mood_model.pkl")
#         prediction = model.predict([features])[0]
#         probs = model.predict_proba([features])[0]
#         labels = model.classes_

#         st.success(f"🎧 Tahmin edilen ruh hali: **{prediction}**")
#         show_pie_chart(probs, labels, prediction)
#     else:
#         st.error("❌ Önce modeli eğitmelisiniz!")
