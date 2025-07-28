import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# 1. Fungsi dan Proses Pembersihan Data (sesuai notebook)
def clean_currency(col):
    """Membersihkan kolom mata uang dari koma dan mengubah ke numerik."""
    return pd.to_numeric(col.astype(str).str.replace(",", "").str.strip(), errors='coerce')

def klasifikasi_pola(persen):
    """Mengklasifikasikan penyerapan anggaran."""
    if persen >= 80:
        return "Tinggi"
    elif persen >= 40:
        return "Sedang"
    else:
        return "Rendah"

# Memuat data
df = pd.read_excel("REALISASI ANGGARAN BELANJA.xlsx")

# Mengganti nama kolom
df.columns = [
    "AKUN", "URAIAN", "ANGGARAN_SEMULA", "ANGGARAN_REVISI", "REALISASI",
    "PENGEMBALIAN", "REALISASI_NETTO", "PERSEN_REALISASI", "SISA_ANGGARAN", "TRIWULAN"
]

# Menghapus baris yang tidak perlu dan mereset index
df = df.iloc[2:].reset_index(drop=True)

# Membersihkan kolom numerik
numeric_cols = ["ANGGARAN_SEMULA", "ANGGARAN_REVISI", "REALISASI",
                "PENGEMBALIAN", "REALISASI_NETTO", "SISA_ANGGARAN", "TRIWULAN"]
for col in numeric_cols:
    df[col] = clean_currency(df[col])

# Menghapus baris dengan nilai AKUN kosong
df = df[df["AKUN"].notna()]

# Menghapus semua baris yang memiliki nilai NaN
df = df.dropna(how='any').reset_index(drop=True)

# 2. Feature Engineering
# Hitung ulang PERSEN_REALISASI untuk memastikan konsistensi
df["PERSEN_REALISASI"] = (df["REALISASI_NETTO"] / df["ANGGARAN_REVISI"]) * 100
df['KATEGORI_SERAPAN'] = df['PERSEN_REALISASI'].apply(klasifikasi_pola)

# 3. Persiapan Training Model
# Definisikan fitur (X) dan target (y)
features = ['AKUN', 'ANGGARAN_SEMULA', 'ANGGARAN_REVISI', 'SISA_ANGGARAN', 'TRIWULAN']
X = df[features]
y = df['KATEGORI_SERAPAN']

# Inisialisasi dan fit StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Definisi dan Training Pipeline
# Gunakan hyperparameter terbaik dari notebook
# {'rf__max_depth': 7, 'rf__min_samples_leaf': 5, 'rf__min_samples_split': 10, 'rf__n_estimators': 50}
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(
        n_estimators=50,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

# Melatih pipeline dengan seluruh data yang sudah diskalakan
pipeline.fit(X_scaled, y)

print("Model pipeline berhasil dilatih.")

# 5. Simpan Model dan Scaler
joblib.dump(pipeline, 'model_pipeline.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model pipeline dan scaler berhasil disimpan ke dalam file.")