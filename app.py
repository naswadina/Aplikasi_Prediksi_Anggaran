import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis & Prediksi Anggaran",
    page_icon="💰",
    layout="wide"
)

# --- FUNGSI LOADING (DENGAN CACHE) ---
@st.cache_data
def load_analytic_data():
    df = pd.read_excel('realisasi_anggaran_bersih.xlsx')
    # Konversi tipe data
    df['TRIWULAN'] = df['TRIWULAN'].astype(str)
    df['AKUN'] = df['AKUN'].astype(str)
    
    # --- [BARU] Membuat kolom 'Jenis Belanja' dari Kode Akun ---
    def get_jenis_belanja(akun):
        prefix = akun[:2]
        if prefix == '51':
            return 'Belanja Pegawai'
        elif prefix == '52':
            return 'Belanja Barang'
        elif prefix == '53':
            return 'Belanja Modal'
        else:
            return 'Lainnya'
    df['JENIS_BELANJA'] = df['AKUN'].apply(get_jenis_belanja)

    # --- [BARU] Menghitung kolom 'Persen Realisasi' untuk visualisasi ---
    # Menggunakan np.divide untuk menghindari error pembagian dengan nol
    df['PERSEN_REALISASI'] = np.divide(df['REALISASI_NETTO'], df['ANGGARAN_REVISI']) * 100
    df['PERSEN_REALISASI'] = df['PERSEN_REALISASI'].fillna(0) # Ganti NaN dengan 0

    return df

# Cache untuk model dan scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('model_pipeline.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        return None, None

# Muat data dan model
df_analytic = load_analytic_data()
model, scaler = load_model_and_scaler()

# --- JUDUL UTAMA APLIKASI ---
st.title("💰 Analisis & Prediksi Penyerapan Anggaran")
st.markdown("Selamat datang di dashboard interaktif anda. Gunakan tab di bawah untuk menjelajahi analisis kinerja anggaran atau untuk membuat prediksi penyerapan.")

# --- MEMBUAT TAB ---
tab1, tab2 = st.tabs(["📈 Dashboard Analitik", "🤖 Aplikasi Prediksi"])

# ==============================================================================
# --- TAB 1: DASHBOARD ANALITIK ---
# ==============================================================================
with tab1:
    st.header("📈 Dashboard Analitik Kinerja Anggaran")
    
    # --- FILTER ---
    st.subheader("Filter Data Analitik")
    
    # --- [BARU] Menambahkan expander untuk filter agar lebih rapi ---
    with st.expander("Buka Panel Filter"):
        col1, col2, col3 = st.columns(3)
        with col1:
            kategori_filter = st.multiselect(
                "Pilih Kategori Serapan:",
                options=df_analytic["KATEGORI_SERAPAN"].unique(),
                default=df_analytic["KATEGORI_SERAPAN"].unique()
            )
        with col2:
            triwulan_filter = st.multiselect(
                "Pilih Triwulan:",
                options=df_analytic["TRIWULAN"].unique(),
                default=df_analytic["TRIWULAN"].unique()
            )
        # --- [BARU] Filter Jenis Belanja ---
        with col3:
            jenis_belanja_filter = st.multiselect(
                "Pilih Jenis Belanja:",
                options=df_analytic["JENIS_BELANJA"].unique(),
                default=df_analytic["JENIS_BELANJA"].unique()
            )

        # --- [BARU] Filter Rentang Anggaran ---
        min_anggaran = int(df_analytic['ANGGARAN_REVISI'].min())
        max_anggaran = int(df_analytic['ANGGARAN_REVISI'].max())
        rentang_anggaran = st.slider(
            "Pilih Rentang Anggaran Revisi (Rp):",
            min_value=min_anggaran,
            max_value=max_anggaran,
            value=(min_anggaran, max_anggaran)
        )

    # Terapkan semua filter ke dataframe
    df_selection = df_analytic.query(
        "KATEGORI_SERAPAN == @kategori_filter & TRIWULAN == @triwulan_filter & JENIS_BELANJA == @jenis_belanja_filter & ANGGARAN_REVISI >= @rentang_anggaran[0] & ANGGARAN_REVISI <= @rentang_anggaran[1]"
    )
    if df_selection.empty:
        st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih. Coba sesuaikan filter Anda.", icon="⚠️")
    else:
        st.markdown("---")

    # --- KPI ---
    total_anggaran = float(df_selection['ANGGARAN_REVISI'].sum())
    total_realisasi = float(df_selection['REALISASI_NETTO'].sum())
    if total_anggaran > 0:
        persen_realisasi = (total_realisasi / total_anggaran) * 100
    else:
        persen_realisasi = 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="💵 Total Anggaran Revisi", value=f"Rp {total_anggaran:,.0f}")
    kpi2.metric(label="🎯 Total Realisasi Netto", value=f"Rp {total_realisasi:,.0f}")
    kpi3.metric(label="📊 Persentase Realisasi", value=f"{persen_realisasi:.2f} %")

    st.markdown("---")

    # --- GRAFIK UTAMA ---
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("🥧 Komposisi Anggaran per Jenis Belanja")
        fig_jenis_belanja = px.pie(
            df_selection, names='JENIS_BELANJA', values='ANGGARAN_REVISI', hole=.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_jenis_belanja, use_container_width=True)

    with chart_right:
        st.subheader("📊 Tren Realisasi per Triwulan")
        df_tren = df_selection.groupby('TRIWULAN')[['ANGGARAN_REVISI', 'REALISASI_NETTO']].sum().reset_index()
        fig_tren = px.bar(
            df_tren, x='TRIWULAN', y=['ANGGARAN_REVISI', 'REALISASI_NETTO'],
            barmode='group', text_auto='.2s'
        )
        st.plotly_chart(fig_tren, use_container_width=True)

    st.markdown("---")
    
    # --- [BARU] VISUALISASI TAMBAHAN ---
    st.subheader("🔍 Analisis Lebih Lanjut")
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        # Scatter Plot untuk melihat sebaran data
        st.markdown("##### Sebaran Realisasi vs Anggaran")
        fig_scatter = px.scatter(
            df_selection,
            x="ANGGARAN_REVISI",
            y="REALISASI_NETTO",
            color="KATEGORI_SERAPAN",
            hover_name="URAIAN",
            log_x=True, # Menggunakan skala logaritmik agar sebaran lebih terlihat
            log_y=True,
            size_max=60
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with vis_col2:
        # Top 10 Sisa Anggaran Terbesar
        st.markdown("##### Top 10 Sisa Anggaran Terbesar")
        df_top10_sisa = df_selection.nlargest(10, 'SISA_ANGGARAN')
        fig_top10 = px.bar(
            df_top10_sisa,
            x="SISA_ANGGARAN",
            y="URAIAN",
            orientation='h',
            text_auto='.2s'
        )
        fig_top10.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top10, use_container_width=True)

    # --- TABEL DETAIL ---
    st.subheader("📋 Detail Data Sesuai Filter")
    st.dataframe(df_selection)

# ==============================================================================
# --- TAB 2: APLIKASI PREDIKSI ---
# ==============================================================================
with tab2:
    st.header("🤖 Aplikasi Prediksi Kategori Penyerapan Anggaran")
    
    if model is None or scaler is None:
        st.error(
            "File `model_pipeline.joblib` atau `scaler.joblib` tidak ditemukan."
        )
    else:
        st.sidebar.header("⚙️ Input Parameter Prediksi")
        akun_pred = st.sidebar.number_input('Kode Akun (Contoh: 511129)', format="%f")
        anggaran_semula_pred = st.sidebar.number_input('Anggaran Semula (Rp)', min_value=0.0, format="%f")
        anggaran_revisi_pred = st.sidebar.number_input('Anggaran Setelah Revisi (Rp)', min_value=0.0, format="%f")
        sisa_anggaran_pred = st.sidebar.number_input('Sisa Anggaran (Rp)', min_value=0.0, format="%f")
        triwulan_pred = st.sidebar.selectbox('Triwulan', [1, 2, 3, 4], key='pred_triwulan')
        
        if st.sidebar.button('🔮 Prediksi', type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                'AKUN': [akun_pred], 'ANGGARAN_SEMULA': [anggaran_semula_pred],
                'ANGGARAN_REVISI': [anggaran_revisi_pred], 'SISA_ANGGARAN': [sisa_anggaran_pred],
                'TRIWULAN': [triwulan_pred]
            })

            st.subheader("➡️ Data Input:")
            st.dataframe(input_data)

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            st.subheader("✅ Hasil Prediksi:")
            kategori_pred = prediction[0]

            if kategori_pred == "Tinggi":
                st.success(f"**Kategori Penyerapan: {kategori_pred}** 📈")
            elif kategori_pred == "Sedang":
                st.warning(f"**Kategori Penyerapan: {kategori_pred}** 📊")
            else:
                st.error(f"**Kategori Penyerapan: {kategori_pred}** 📉")

            st.subheader("🔢 Probabilitas Prediksi:")
            proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=["Probabilitas"])
            st.dataframe(proba_df.style.format("{:.2%}"))
        
        st.info(
            "**Cara Penggunaan:**\n"
            "1. Isi semua parameter pada panel di sebelah kiri.\n"
            "2. Klik tombol '🔮 Prediksi'.\n"
            "3. Hasil prediksi akan muncul di halaman ini.",
            icon="💡"
        )
        st.markdown(
                f"<div style='text-align: center; padding: 50px;'>"
                f"<span style='font-size: 100px;'>🤖</span>"
                f"<h3>Menunggu Input Anda...</h3>"
                f"</div>",
                unsafe_allow_html=True
        )
