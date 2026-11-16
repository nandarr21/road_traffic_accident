from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # simpan grafik ke file tanpa membuka jendela
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------
# Konfigurasi path (relatif terhadap lokasi file ini)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "RTA Dataset.csv"
EDA_DIR = BASE_DIR / "reports" / "eda"           # grafik untuk laporan
STATIC_IMG_DIR = BASE_DIR / "public" / "static" / "images"  # grafik untuk dashboard web

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

# Urutan kelas dari paling ringan ke paling parah, dan warnanya
CLASS_ORDER = ["Slight Injury", "Serious Injury", "Fatal injury"]
CLASS_COLORS = {"Slight Injury": "#3b82f6",   # biru
                "Serious Injury": "#f59e0b",  # oranye
                "Fatal injury": "#ef4444"}    # merah


def garis(judul):
    """Mencetak judul bagian agar output terminal mudah dibaca."""
    print("\n" + "=" * 70)
    print(judul)
    print("=" * 70)


def info_dasar(df):
    """Lima data pertama, ukuran data, dan tipe data."""
    garis("1. LIMA DATA PERTAMA (11 fitur + target)")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[FEATURES + [TARGET]].head())

    garis("2. UKURAN DATA")
    print(f"Jumlah baris : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]} (yang dipakai: {len(FEATURES)} fitur + 1 target)")

    garis("3. TIPE DATA")
    print(df[FEATURES + [TARGET]].dtypes.astype(str).to_string())


def cek_missing(df):
    """Hitung NaN dan 'Unknown' per fitur, lalu buat grafiknya."""
    garis("4. MISSING VALUE")
    tabel = pd.DataFrame({
        "NaN": df[FEATURES].isna().sum(),
        "Unknown": df[FEATURES].apply(lambda s: s.astype(str).str.lower().eq("unknown").sum()),
    })
    tabel["NaN_%"] = (tabel["NaN"] / len(df) * 100).round(2)
    print(tabel.to_string())
    print(f"\nMissing pada target: {df[TARGET].isna().sum()}")

    # Grafik batang bertumpuk: NaN + Unknown
    data = tabel[["NaN", "Unknown"]].sort_values("NaN")
    ax = data.plot(kind="barh", stacked=True, figsize=(9, 5),
                   color=["#ef4444", "#94a3b8"])
    ax.set_title("Missing Value (NaN) dan Nilai 'Unknown' per Fitur")
    ax.set_xlabel("Jumlah baris")
    ax.set_ylabel("")
    plt.tight_layout()
    simpan(ax.figure, "02_missing_value.png")


def plot_distribusi_target(df):
    """Countplot target dengan label jumlah dan persentase."""
    garis("5. DISTRIBUSI TARGET")
    jumlah = df[TARGET].value_counts().reindex(CLASS_ORDER)
    persen = (jumlah / jumlah.sum() * 100).round(2)
    print(pd.DataFrame({"jumlah": jumlah, "persen (%)": persen}).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(x=TARGET, data=df, order=CLASS_ORDER, hue=TARGET,
                  palette=CLASS_COLORS, legend=False, ax=ax)
    # Tulis jumlah dan persentase di atas setiap batang
    for i, kelas in enumerate(CLASS_ORDER):
        ax.text(i, jumlah[kelas] + 150, f"{jumlah[kelas]:,}\n({persen[kelas]}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Distribusi Tingkat Keparahan Kecelakaan")
    ax.set_xlabel("Accident_severity")
    ax.set_ylabel("Jumlah kecelakaan")
    ax.set_ylim(0, jumlah.max() * 1.18)
    plt.tight_layout()
    simpan(fig, "01_distribusi_target.png")
    # Salinan untuk dashboard aplikasi web
    fig.savefig(STATIC_IMG_DIR / "distribusi_target.png", dpi=120)
    plt.close(fig)


def plot_fitur_vs_target(df, kolom, nama_file):
    """
    Grafik proporsi tingkat keparahan untuk setiap kategori fitur.

    Karena kelas sangat tidak seimbang (Slight Injury 84%), grafik jumlah
    biasa akan didominasi warna biru. Maka dipakai PROPORSI (100% stacked),
    sehingga perbedaan persentase Serious/Fatal antar kategori terlihat.
    """
    data = df[[kolom, TARGET]].copy()
    data[kolom] = data[kolom].fillna("NaN (kosong)").astype(str)

    proporsi = pd.crosstab(data[kolom], data[TARGET], normalize="index")
    proporsi = proporsi.reindex(columns=CLASS_ORDER, fill_value=0) * 100
    jumlah = data[kolom].value_counts()
    # Urutkan dari persentase Serious+Fatal tertinggi
    proporsi = proporsi.loc[(proporsi["Serious Injury"] + proporsi["Fatal injury"])
                            .sort_values().index]

    tinggi = max(3.5, 0.45 * len(proporsi) + 1.5)
    fig, ax = plt.subplots(figsize=(10, tinggi))
    proporsi.plot(kind="barh", stacked=True, ax=ax,
                  color=[CLASS_COLORS[k] for k in CLASS_ORDER], width=0.75)
    # Label jumlah data (n) di sebelah kanan setiap batang
    for i, kategori in enumerate(proporsi.index):
        ax.text(101, i, f"n={jumlah[kategori]:,}", va="center", fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Persentase (%)")
    ax.set_ylabel("")
    ax.set_title(f"Proporsi Tingkat Keparahan menurut {kolom}")
    ax.legend(title="Accident_severity", loc="upper center",
              bbox_to_anchor=(0.5, -0.12 if len(proporsi) > 6 else -0.25), ncol=3)
    plt.tight_layout()
    simpan(fig, nama_file)

    # Ringkasan di terminal: persentase Serious + Fatal per kategori
    ringkas = proporsi[["Serious Injury", "Fatal injury"]].round(2)
    ringkas["n"] = jumlah.reindex(ringkas.index)
    return ringkas.sort_values("Fatal injury", ascending=False)


def plot_hubungan_fitur(df):
    """Buat grafik hubungan untuk ke-11 fitur dan cetak ringkasannya."""
    garis("6. HUBUNGAN FITUR DENGAN TARGET (persentase per kategori)")
    for i, kolom in enumerate(FEATURES, start=1):
        ringkas = plot_fitur_vs_target(df, kolom, f"03_{i:02d}_{kolom}.png")
        print(f"\n--- {kolom} (diurutkan dari % Fatal tertinggi) ---")
        print(ringkas.to_string())


def simpan(fig, nama_file):
    """Simpan grafik ke folder reports/eda lalu tutup figurnya."""
    path = EDA_DIR / nama_file
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[grafik disimpan] {path.relative_to(BASE_DIR)}")


def main():
    if not DATA_PATH.exists():
        print(f"File dataset tidak ditemukan: {DATA_PATH}")
        return
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(DATA_PATH)
    info_dasar(df)
    cek_missing(df)
    plot_distribusi_target(df)
    plot_hubungan_fitur(df)

    garis("SELESAI")
    print(f"Semua grafik EDA tersimpan di: {EDA_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
