import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Load model ===
with open("D:/Bootcamp ML & AI/admission_app/hasil.pkl", "rb") as file:
    model = pickle.load(file)

# === Min-Max values dari data sebelum scaling ===
feature_min = {
    'motiv_letter_strength': 1,
    'recommendation_strength': 1,
    'gpa': 6.8,
    'research_exp': 0,   # 0 = tidak punya pengalaman riset
    'univ_tier': 0       # 0 = low, 1 = high
}

feature_max = {
    'motiv_letter_strength': 5,
    'recommendation_strength': 5,
    'gpa': 9.92,
    'research_exp': 1,
    'univ_tier': 1
}

# === Fungsi scaling manual ===
def minmax_scale(value, feature):
    min_val = feature_min[feature]
    max_val = feature_max[feature]
    return (value - min_val) / (max_val - min_val)

# === Aplikasi Streamlit ===
st.title("🎓 Prediksi Penerimaan Mahasiswa")
st.markdown("""
Aplikasi ini digunakan untuk **memprediksi peluang diterima/tidaknya calon mahasiswa**
berdasarkan data akademik dan non-akademik seperti IPK, kekuatan surat motivasi,
surat rekomendasi, pengalaman riset, dan tier universitas asal.

---

### 🧠 Tujuan Aplikasi
Membantu calon mahasiswa untuk:
- Mengetahui seberapa besar peluang diterima berdasarkan profil mereka  
- Melihat bagaimana faktor-faktor tertentu (IPK, surat rekomendasi, dll) memengaruhi hasil seleksi

---

### 🗂️ Data Training
Dataset yang digunakan adalah data admission dengan fitur-fitur yang digunakan :
- **motiv_letter_strength** : kekuatan surat motivasi (1–5)  
- **recommendation_strength** : kekuatan surat rekomendasi (1–5)  
- **gpa** : IPK mahasiswa (6.8–9.92 pada data mentah, diubah ke skala 0–1 saat training)  
- **research_exp** : pengalaman riset (0 = tidak, 1 = ya)  
- **univ_tier** : tingkat universitas asal (0 = low, 1 = high)

---

### ⚙️ Model yang Digunakan
Model ini menggunakan **Lasso Regression (LassoCV)** dengan parameter terbaik hasil
cross-validation pada 5 fold.

Model dilatih menggunakan data yang sudah melalui proses:
- **Pembersihan missing value & duplikasi**
- **Encoding kategorikal (ya/tidak, high/low)**
- **Penskalaan dengan Min-Max Scaler**

---

""")
# =============================
# Sidebar Input
# =============================
st.sidebar.header("🧾 Formulir Prediksi")
st.sidebar.write("Masukkan data calon mahasiswa di bawah ini:")

motiv_letter_strength = st.sidebar.slider("Kekuatan Surat Motivasi (1–5)", 1, 5, 3)
recommendation_strength = st.sidebar.slider("Kekuatan Surat Rekomendasi (1–5)", 1, 5, 3)
gpa = st.sidebar.number_input("GPA (0.0–10.0)", min_value=0.0, max_value=10.0, value=3.0, step=0.01)
research_exp = st.sidebar.selectbox("Pengalaman Riset", ["Tidak", "Ya"])
univ_tier = st.sidebar.selectbox("Tingkat Universitas Asal", ["Low", "High"])

# Konversi kategori ke numerik
research_exp_val = 1 if research_exp == "Ya" else 0
univ_tier_val = 1 if univ_tier == "High" else 0

# === Scaling input sesuai model ===
data_input_scaled = pd.DataFrame({
    'motiv_letter_strength': [minmax_scale(motiv_letter_strength, 'motiv_letter_strength')],
    'recommendation_strength': [minmax_scale(recommendation_strength, 'recommendation_strength')],
    'gpa': [minmax_scale(gpa, 'gpa')],
    'research_exp': [minmax_scale(research_exp_val, 'research_exp')],
    'univ_tier': [minmax_scale(univ_tier_val, 'univ_tier')]
})

# === Prediksi ===
if st.button("Prediksi"):
    # pastikan semua input sudah float
    data_input_scaled = data_input_scaled.astype(float)
    
    # prediksi
    pred = model.predict(data_input_scaled)[0]
    
    # pastikan hasil prediksi float
    pred = float(pred)
    
    st.write(f"**Nilai prediksi (probabilitas diterima): {pred:.3f}**")
    
    # bandingkan dengan ambang batas
    if pred >= 0.5:
        st.success("🎉 Mahasiswa **berpeluang diterima!**")
    else:
        st.error("❌ Mahasiswa **kemungkinan tidak diterima.**")

    st.markdown("""
    ---
    #### 📘 Catatan
    Nilai prediksi menunjukkan estimasi seberapa besar peluang calon mahasiswa diterima.
    Model ini tidak menjamin hasil keputusan akhir penerimaan, melainkan memberikan perkiraan berbasis data historis.
    """)
else:
    st.info("Gunakan panel di **sidebar kiri** untuk mengisi data calon mahasiswa dan tekan tombol *Prediksi*.")
