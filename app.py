import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Dashboard Peramalan PO", layout="wide")
st.title("📊 Dashboard Peramalan Jumlah PO Konveksi")
st.write("Upload file rekap data PO untuk melihat proyeksi produksi bulan berikutnya.")

# 2. Fitur Upload File
uploaded_file = st.file_uploader("Upload file Excel (.xlsx) atau CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Membaca data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("1. Data Historis PO")
    st.dataframe(df.head(10)) # Menampilkan 10 baris pertama

    # 3. Pengaturan Parameter di Sidebar (Panel Samping)
    st.sidebar.header("Pengaturan Peramalan")
    
    # Pilih kolom produk (mengabaikan kolom Tahun, Bulan, Periode)
    kolom_non_produk = ['Tahun', 'Bulan', 'Periode']
    daftar_produk = [col for col in df.columns if col not in kolom_non_produk]
    
    produk_pilihan = st.sidebar.multiselect(
        "Pilih Produk untuk Diramal (Disarankan Top 3):",
        options=daftar_produk,
        default=daftar_produk[:3] # Otomatis memilih 3 produk pertama
    )
    
    bulan_prediksi = st.sidebar.slider("Jumlah Periode ke Depan yang Diramal:", 1, 12, 3)

    # 4. Proses Perhitungan Regresi Linear
    if produk_pilihan:
        st.subheader("2. Hasil Peramalan (Regresi Linear)")
        
        # Pastikan data historis hanya mengambil baris yang sudah ada angka Periodenya
        df_valid = df.dropna(subset=['Periode']).copy()
        X = df_valid[['Periode']].values
        periode_terakhir = int(df_valid['Periode'].max())
        
        # Membuat array periode masa depan
        periode_depan = np.arange(periode_terakhir + 1, periode_terakhir + 1 + bulan_prediksi).reshape(-1, 1)
        
        hasil_forecast = pd.DataFrame({'Periode': periode_depan.flatten()})
        
        for produk in produk_pilihan:
            # Mengisi nilai kosong dengan 0 untuk pelatihan model
            y = df_valid[produk].fillna(0).values
            
            model = LinearRegression()
            model.fit(X, y)
            
            prediksi = model.predict(periode_depan)
            # Membulatkan hasil & mencegah angka negatif (minimal 0)
            prediksi_bersih = np.maximum(0, np.round(prediksi)).astype(int)
            hasil_forecast[produk] = prediksi_bersih

        # Menampilkan Tabel Hasil
        st.dataframe(hasil_forecast, use_container_width=True)

        # 5. Menampilkan Grafik Tren Gabungan
        st.subheader("3. Grafik Tren Historis & Proyeksi")
        df_gabungan = pd.concat([
            df_valid[['Periode'] + produk_pilihan],
            hasil_forecast
        ], ignore_index=True).set_index('Periode')
        
        st.line_chart(df_gabungan)
        
        # 6. Tombol Download Hasil ke CSV
        csv_hasil = hasil_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Tabel Hasil Peramalan (CSV)",
            data=csv_hasil,
            file_name="hasil_peramalan_po.csv",
            mime="text/csv"
        )
