import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Penyerapan Anggaran", page_icon="📊")

# --- Load Model and Scaler ---
try:
    model = joblib.load('https://raw.githubusercontent.com/naswadina/Aplikasi_Prediksi_Anggaran/blob/main/model_pipeline.joblib')
    scaler = joblib.load('https://raw.githubusercontent.com/naswadina/Aplikasi_Prediksi_Anggaran/blob/main/scaler.joblib')
except FileNotFoundError:
    st.error(
        "Model files not found! Please run the `train_and_save_model.py` script"
        "first to generate 'model_pipeline.joblib' and 'scaler.joblib'."
    )
    st.stop()

# --- User Interface ---
st.title("📊 Aplikasi Prediksi Kategori Penyerapan Anggaran")
st.markdown(
    "Masukkan data anggaran di bawah ini untuk memprediksi kategori penyerapannya "
    "(Tinggi, Sedang, atau Rendah)."
)

st.sidebar.header("Input Parameter Anggaran")

# Create input fields
akun = st.sidebar.number_input('Kode Akun (Contoh: 511129)', format="%f")
anggaran_semula = st.sidebar.number_input('Anggaran Semula (Rp)', min_value=0.0, format="%f")
anggaran_revisi = st.sidebar.number_input('Anggaran Setelah Revisi (Rp)', min_value=0.0, format="%f")
sisa_anggaran = st.sidebar.number_input('Sisa Anggaran (Rp)', min_value=0.0, format="%f")
triwulan = st.sidebar.selectbox('Triwulan', [1, 2, 3, 4])

# --- Prediction Logic ---
if st.sidebar.button('Prediksi Kategori Penyerapan', type="primary"):
    input_data = pd.DataFrame({
        'AKUN': [akun],
        'ANGGARAN_SEMULA': [anggaran_semula],
        'ANGGARAN_REVISI': [anggaran_revisi],
        'SISA_ANGGARAN': [sisa_anggaran],
        'TRIWULAN': [triwulan]
    })

    st.subheader("Data Input:")
    st.dataframe(input_data)

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Display result
    st.subheader("Hasil Prediksi:")
    kategori = prediction[0]

    if kategori == "Tinggi":
        st.success(f"**Kategori Penyerapan: {kategori}**")
    elif kategori == "Sedang":
        st.warning(f"**Kategori Penyerapan: {kategori}**")
    else: # Rendah
        st.error(f"**Kategori Penyerapan: {kategori}**")

    st.subheader("Probabilitas Prediksi:")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=["Probabilitas"])
    st.dataframe(proba_df.style.format("{:.2%}"))

st.info(
    "**Cara Penggunaan:**\n"
    "1. Isi semua parameter pada panel di sebelah kiri.\n"
    "2. Klik tombol 'Prediksi Kategori Penyerapan'.\n"
    "3. Hasil prediksi akan muncul di halaman ini.",
    icon="💡"
)
