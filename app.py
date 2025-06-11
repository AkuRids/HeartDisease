import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('heart_model.pkl')

st.title("Prediksi Penyakit Jantung")

# Form input (fitur berdasarkan heart.csv)
with st.form("form_heart"):
    age = st.number_input('Usia', min_value=1, max_value=120)
    sex = st.selectbox('Jenis Kelamin (0: Perempuan, 1: Laki-laki)', [0, 1])
    cp = st.selectbox('Tipe Nyeri Dada (0-3)', [0, 1, 2, 3])
    trestbps = st.number_input('Tekanan Darah Istirahat', min_value=80, max_value=200)
    chol = st.number_input('Kolesterol', min_value=100, max_value=600)
    fbs = st.selectbox('Gula Darah > 120 mg/dl (1 = ya, 0 = tidak)', [0, 1])
    restecg = st.selectbox('Hasil EKG Saat Istirahat (0-2)', [0, 1, 2])
    thalach = st.number_input('Detak Jantung Maksimum', min_value=50, max_value=250)
    exang = st.selectbox('Angina yang Diinduksi Olahraga (1 = ya, 0 = tidak)', [0, 1])
    oldpeak = st.number_input('Oldpeak (Depresi ST)', min_value=0.0, max_value=6.0, step=0.1)
    slope = st.selectbox('Slope dari ST Segment (0-2)', [0, 1, 2])
    ca = st.selectbox('Jumlah Pembuluh Terdeteksi (0-4)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thal (0 = normal, 1 = fixed defect, 2 = reversable)', [0, 1, 2])

    submit = st.form_submit_button("Prediksi")

# Ketika tombol ditekan
if submit:
    # Format input ke bentuk array sesuai urutan pelatihan model
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]

    # Tampilkan hasil
    if prediction == 1:
        st.error("Hasil: Berisiko Penyakit Jantung")
    else:
        st.success("Hasil: Tidak Berisiko Penyakit Jantung")
