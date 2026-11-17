
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Peramalan PO",
    layout="wide"
)

st.title("📊 Dashboard Peramalan Jumlah PO Konveksi")
st.write(
    "Upload file rekap data PO untuk melihat "
    "proyeksi produksi bulan berikutnya."
)

# 2. Upload file
uploaded_file = st.file_uploader(
    "Upload file Excel (.xlsx) atau CSV",
    type=["xlsx", "csv"]
)

if uploaded_file is not None:

    # Membaca data
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Membersihkan nama kolom
        df.columns = df.columns.astype(str).str.strip()

        st.subheader("1. Data Historis PO")
        st.dataframe(df, use_container_width=True)

        # Memeriksa kolom wajib
        kolom_wajib = ["Tahun", "Bulan"]

        if not all(k in df.columns for k in kolom_wajib):
            st.error("Kolom Tahun dan Bulan harus tersedia.")
            st.stop()

        # 3. Membuat urutan periode berdasarkan tahun dan bulan
        nama_bulan = {
            "januari": 1, "februari": 2,
            "maret": 3, "april": 4,
            "mei": 5, "juni": 6,
            "juli": 7, "agustus": 8,
            "september": 9, "oktober": 10,
            "november": 11, "desember": 12
        }

        df["Tahun"] = pd.to_numeric(
            df["Tahun"], errors="coerce"
        )

        df["Bulan_Angka"] = (
            df["Bulan"].astype(str).str.strip().str.lower()
            .map(nama_bulan)
        )

        # Jika bulan sudah berbentuk angka
        bulan_numerik = pd.to_numeric(
            df["Bulan"], errors="coerce"
        )

        df["Bulan_Angka"] = df["Bulan_Angka"].fillna(
            bulan_numerik
        )

        df = df.dropna(
            subset=["Tahun", "Bulan_Angka"]
        ).copy()

        df["Tahun"] = df["Tahun"].astype(int)
        df["Bulan_Angka"] = df["Bulan_Angka"].astype(int)

        # Memastikan nomor bulan valid
        df = df[
            df["Bulan_Angka"].between(1, 12)
        ].copy()

        # Mengurutkan data
        df = df.sort_values(
            ["Tahun", "Bulan_Angka"]
        ).reset_index(drop=True)

        # Membuat periode berkelanjutan
        periode_awal = (
            df["Tahun"].min() * 12
            + df["Bulan_Angka"].min() - 1
        )

        df["Periode"] = (
            df["Tahun"] * 12
            + df["Bulan_Angka"] - 1
            - periode_awal + 1
        )

        # 4. Memilih produk
        kolom_non_produk = [
            "Tahun", "Bulan", "Bulan_Angka",
            "Periode"
        ]

        daftar_produk = [
            col for col in df.columns
            if col not in kolom_non_produk
        ]

        st.sidebar.header("Pengaturan Peramalan")

        produk_pilihan = st.sidebar.multiselect(
            "Pilih Produk untuk Diramal:",
            options=daftar_produk,
            default=daftar_produk[:3]
        )

        bulan_prediksi = st.sidebar.slider(
            "Jumlah Periode ke Depan yang Diramal:",
            1, 12, 3
        )

        # 5. Proses regresi linear
        if produk_pilihan:

            st.subheader(
                "2. Hasil Peramalan (Regresi Linear)"
            )

            X = df[["Periode"]].values

            periode_terakhir = int(
                df["Periode"].max()
            )

            periode_depan = np.arange(
                periode_terakhir + 1,
                periode_terakhir + 1 + bulan_prediksi
            ).reshape(-1, 1)

            hasil_forecast = pd.DataFrame({
                "Periode": periode_depan.flatten()
            })

            hasil_detail = []

            for produk in produk_pilihan:

                # Mengubah data produk menjadi numerik
                y_data = pd.to_numeric(
                    df[produk], errors="coerce"
                )

                # Menggunakan hanya pasangan data valid
                data_model = pd.DataFrame({
                    "Periode": df["Periode"],
                    "Jumlah": y_data
                }).dropna()

                if len(data_model) < 2:
                    st.warning(
                        f"Data {produk} kurang dari 2 "
                        "periode valid. Tidak dapat "
                        "dilakukan regresi."
                    )
                    continue

                X_train = data_model[
                    ["Periode"]
                ].values

                y_train = data_model["Jumlah"].values

                model = LinearRegression()
                model.fit(X_train, y_train)

                prediksi = model.predict(periode_depan)

                # Menampilkan hasil prediksi asli
                with st.expander(
                    f"Detail perhitungan {produk}"
                ):
                    st.write(
                        "Jumlah data yang digunakan:",
                        len(data_model)
                    )
                    st.write(
                        "Persamaan regresi:",
                        f"Y = {model.intercept_:.2f} "
                        f"+ ({model.coef_[0]:.2f} × X)"
                    )
                    st.write(
                        "Prediksi sebelum pembulatan:",
                        prediksi
                    )

                # Membulatkan dan membatasi nilai minimum 0
                prediksi_bersih = np.maximum(
                    0, np.round(prediksi)
                ).astype(int)

                hasil_forecast[produk] = prediksi_bersih

                hasil_detail.append(produk)

            # 6. Menampilkan hasil
            if hasil_detail:

                st.dataframe(
                    hasil_forecast,
                    use_container_width=True
                )

                # 7. Grafik historis dan prediksi
                st.subheader(
                    "3. Grafik Tren Historis & Proyeksi"
                )

                df_gabungan = pd.concat([
                    df[["Periode"] + hasil_detail],
                    hasil_forecast[
                        ["Periode"] + hasil_detail
                    ]
                ], ignore_index=True)

                df_gabungan = df_gabungan.set_index(
                    "Periode"
                )

                st.line_chart(df_gabungan)

                # 8. Download hasil
                csv_hasil = hasil_forecast[
                    ["Periode"] + hasil_detail
                ].to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Hasil Peramalan",
                    data=csv_hasil,
                    file_name="hasil_peramalan_po.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "Tidak ada produk yang memiliki "
                    "cukup data untuk diramal."
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
