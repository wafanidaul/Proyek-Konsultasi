import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Dashboard Peramalan PO",
    layout="wide"
)

st.title("📊 Dashboard Peramalan Jumlah PO Konveksi")
st.write("Upload data rekap PO untuk melakukan peramalan jumlah PO.")

# ==========================================
# 2. UPLOAD DATA
# ==========================================

uploaded_file = st.file_uploader(
    "Upload file Excel atau CSV",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.astype(str).str.strip()

    st.subheader("1. Data Historis PO")
    st.dataframe(df, use_container_width=True)

    # ==========================================
    # 3. PEMERIKSAAN DATA
    # ==========================================

    if not {"Tahun", "Bulan"}.issubset(df.columns):
        st.error("Kolom Tahun dan Bulan wajib tersedia.")
        st.stop()

    df["Tahun"] = pd.to_numeric(
        df["Tahun"], errors="coerce"
    )

    nama_bulan = {
        "januari": 1,
        "februari": 2,
        "maret": 3,
        "april": 4,
        "mei": 5,
        "juni": 6,
        "juli": 7,
        "agustus": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "desember": 12
    }

    bulan_teks = (
        df["Bulan"].astype(str).str.strip().str.lower()
    )

    bulan_angka = pd.to_numeric(
        df["Bulan"], errors="coerce"
    )

    df["Bulan_Angka"] = bulan_teks.map(nama_bulan)
    df["Bulan_Angka"] = df["Bulan_Angka"].fillna(
        bulan_angka
    )

    df = df.dropna(
        subset=["Tahun", "Bulan_Angka"]
    ).copy()

    df["Tahun"] = df["Tahun"].astype(int)
    df["Bulan_Angka"] = df["Bulan_Angka"].astype(int)

    df = df[
        df["Bulan_Angka"].between(1, 12)
    ].copy()

    df = df.sort_values(
        ["Tahun", "Bulan_Angka"]
    ).reset_index(drop=True)

    if df.empty:
        st.error("Tidak ada data tahun dan bulan yang valid.")
        st.stop()

    # ==========================================
    # 4. MEMBUAT PERIODE BERKELANJUTAN
    # ==========================================

    df["Periode"] = (
        (df["Tahun"] - df["Tahun"].min()) * 12
        + df["Bulan_Angka"]
    )

    # ==========================================
    # 5. MEMILIH PRODUK
    # ==========================================

    kolom_non_produk = [
        "Tahun",
        "Bulan",
        "Bulan_Angka",
        "Periode"
    ]

    daftar_produk = [
        kolom for kolom in df.columns
        if kolom not in kolom_non_produk
    ]

    st.sidebar.header("Pengaturan Peramalan")

    produk_pilihan = st.sidebar.multiselect(
        "Pilih Produk untuk Diramal:",
        options=daftar_produk,
        default=daftar_produk[:3]
    )

    bulan_prediksi = st.sidebar.slider(
        "Jumlah Periode ke Depan:",
        min_value=1,
        max_value=12,
        value=3
    )

    # ==========================================
    # 6. REGRESI LINEAR
    # ==========================================

    if produk_pilihan:

        st.subheader("2. Hasil Peramalan Regresi Linear")

        periode_terakhir = int(df["Periode"].max())

        periode_depan = np.arange(
            periode_terakhir + 1,
            periode_terakhir + bulan_prediksi + 1
        ).reshape(-1, 1)

        hasil_forecast = pd.DataFrame({
            "Periode": periode_depan.flatten()
        })

        semua_model = {}

        for produk in produk_pilihan:

            # Mengubah data menjadi numerik
            nilai_produk = pd.to_numeric(
                df[produk],
                errors="coerce"
            )

            data_latih = pd.DataFrame({
                "Periode": df["Periode"],
                "Jumlah_PO": nilai_produk
            })

            # Mengambil data yang benar-benar tersedia
            data_latih = data_latih.dropna()

            if len(data_latih) < 2:
                st.warning(
                    f"Data {produk} tidak cukup untuk regresi."
                )
                continue

            X = data_latih[["Periode"]].values
            Y = data_latih["Jumlah_PO"].values

            # Membentuk model regresi
            model = LinearRegression()
            model.fit(X, Y)

            # Menghitung prediksi
            prediksi = model.predict(periode_depan)

            # Menyimpan model
            semua_model[produk] = model

            # Menampilkan hasil perhitungan
            with st.expander(
                f"Detail Perhitungan {produk}"
            ):

                st.write("Jumlah data yang digunakan:")
                st.write(len(data_latih))

                st.write("Data historis:")
                st.dataframe(data_latih)

                st.write(
                    f"Persamaan regresi: "
                    f"Y = {model.intercept_:.4f} "
                    f"+ ({model.coef_[0]:.4f} × X)"
                )

                st.write("Hasil prediksi asli:")
                st.write(prediksi)

            # Membatasi prediksi negatif menjadi nol
            prediksi_akhir = np.maximum(
                0, np.round(prediksi)
            ).astype(int)

            hasil_forecast[produk] = prediksi_akhir

        # ==========================================
        # 7. TABEL HASIL PERAMALAN
        # ==========================================

        st.subheader("3. Tabel Hasil Peramalan")

        kolom_hasil = [
            p for p in produk_pilihan
            if p in hasil_forecast.columns
        ]

        if kolom_hasil:

            st.dataframe(
                hasil_forecast[
                    ["Periode"] + kolom_hasil
                ],
                use_container_width=True
            )

            # ==========================================
            # 8. GRAFIK HISTORIS DAN PERAMALAN
            # ==========================================

            st.subheader("4. Grafik Tren Historis dan Peramalan")

            for produk in kolom_hasil:

                st.write(f"**Grafik Produk {produk}**")

                historis = pd.DataFrame({
                    "Periode": df["Periode"],
                    "Historis": pd.to_numeric(
                        df[produk], errors="coerce"
                    )
                }).dropna()

                prediksi_grafik = pd.DataFrame({
                    "Periode": hasil_forecast["Periode"],
                    "Peramalan": hasil_forecast[produk]
                })

                grafik = pd.concat([
                    historis.set_index("Periode")["Historis"],
                    prediksi_grafik.set_index("Periode")["Peramalan"]
                ], axis=1)

                st.line_chart(grafik)

            # ==========================================
            # 9. DOWNLOAD HASIL
            # ==========================================

            csv_hasil = hasil_forecast[
                ["Periode"] + kolom_hasil
            ].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Hasil Peramalan",
                data=csv_hasil,
                file_name="hasil_peramalan_po.csv",
                mime="text/csv"
            )

        else:
            st.warning("Tidak ada produk yang berhasil diramal.")

    else:
        st.info("Silakan pilih produk pada sidebar.")

