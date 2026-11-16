from pathlib import Path

import pandas as pd

# Path relatif terhadap lokasi file ini, supaya proyek mudah dipindahkan
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "RTA Dataset.csv"

# 11 fitur (X) dan target (y) sesuai rancangan penelitian
FEATURES = [
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Road_surface_conditions",
    "Weather_conditions",
    "Light_conditions",
    "Type_of_collision",
    "Road_surface_type",
    "Types_of_Junction",
    "Number_of_vehicles_involved",
]
TARGET = "Accident_severity"


def garis(judul):
    """Mencetak judul bagian agar output mudah dibaca."""
    print("\n" + "=" * 70)
    print(judul)
    print("=" * 70)


def cek_kolom(df):
    """Memastikan semua fitur dan target ada di dataset."""
    garis("1. UKURAN DATA DAN KELENGKAPAN KOLOM")
    print(f"Jumlah baris : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]}")
    hilang = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if hilang:
        print("PERINGATAN - kolom tidak ditemukan:", hilang)
    else:
        print("OK - 11 fitur dan kolom target ditemukan semua.")


def cek_target(df):
    """Menampilkan jumlah dan persentase tiap kelas target."""
    garis("2. DISTRIBUSI TARGET (Accident_severity)")
    jumlah = df[TARGET].value_counts()
    persen = (df[TARGET].value_counts(normalize=True) * 100).round(2)
    print(pd.DataFrame({"jumlah": jumlah, "persen (%)": persen}).to_string())


def cek_fitur(df):
    """Menampilkan tipe data, missing value, dan kategori setiap fitur."""
    garis("3. RINGKASAN 11 FITUR")
    ringkas = pd.DataFrame({
        "tipe_data": df[FEATURES].dtypes.astype(str),
        "missing": df[FEATURES].isna().sum(),
        "missing_%": (df[FEATURES].isna().mean() * 100).round(2),
        "jumlah_kategori": df[FEATURES].nunique(),
    })
    print(ringkas.to_string())

    garis("4. DAFTAR KATEGORI PER FITUR (termasuk NaN)")
    for kolom in FEATURES:
        print(f"\n--- {kolom} ---")
        print(df[kolom].value_counts(dropna=False).to_string())


def cek_masalah(df):
    """Mendeteksi masalah data yang perlu diputuskan sebelum training."""
    garis("5. POTENSI MASALAH DATA")

    # a) Nilai 'Unknown' tersembunyi (bukan NaN, tetapi sebenarnya tidak diketahui)
    print("a) Nilai 'Unknown'/'unknown' (missing tersembunyi):")
    for kolom in FEATURES:
        nilai = df[kolom].astype(str)
        n = nilai.str.lower().eq("unknown").sum()
        if n:
            print(f"   {kolom:28s}: {n} baris")

    # b) Duplikat dan label bertentangan pada 11 fitur yang dipakai
    sub = df[FEATURES + [TARGET]].fillna("NaN")
    print(f"\nb) Baris duplikat (11 fitur + target sama persis): {sub.duplicated().sum()}")
    label_per_kombinasi = sub.groupby(FEATURES)[TARGET].nunique()
    konflik = label_per_kombinasi[label_per_kombinasi > 1]
    print(f"   Kombinasi fitur unik            : {len(label_per_kombinasi)}")
    print(f"   Kombinasi dengan label berbeda  : {len(konflik)}")
    print("   (Fitur sama persis tetapi tingkat keparahan berbeda -> "
          "model mana pun tidak bisa 100% benar pada baris ini.)")

    # c) Perkiraan jumlah kelas minoritas di data uji (test_size=0.2, stratify)
    n_fatal = (df[TARGET] == "Fatal injury").sum()
    print(f"\nc) Perkiraan 'Fatal injury' di data uji (20%): sekitar {round(n_fatal * 0.2)} baris")


def main():
    if not DATA_PATH.exists():
        print(f"File dataset tidak ditemukan: {DATA_PATH}")
        print("Pastikan 'RTA Dataset.csv' ada di folder yang sama dengan script ini.")
        return
    df = pd.read_csv(DATA_PATH)
    cek_kolom(df)
    cek_target(df)
    cek_fitur(df)
    cek_masalah(df)


if __name__ == "__main__":
    main()
