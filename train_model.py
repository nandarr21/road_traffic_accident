import json
import os
import shutil
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # simpan grafik ke file tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "RTA Dataset.csv"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"
# Gambar untuk web disimpan di public/static/images (Vercel menyajikan folder public/ lewat CDN)
STATIC_IMG_DIR = BASE_DIR / "public" / "static" / "images"

CATEGORICAL_FEATURES = [
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
]
NUMERIC_FEATURES = ["Number_of_vehicles_involved"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Accident_severity"
CLASS_ORDER = ["Slight Injury", "Serious Injury", "Fatal injury"]
FATAL = "Fatal injury"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# Jumlah proses paralel untuk GridSearchCV. -1 = semua inti CPU.
# Jika laptop terasa berat, ubah menjadi 1 atau 2.
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

MODEL_A = "Model A (Unpruned)"
MODEL_B = "Model B (Pruned, max_depth=4)"
MODEL_C = "Model C (Tuned, GridSearchCV)"
SLUG = {MODEL_A: "model_a", MODEL_B: "model_b", MODEL_C: "model_c"}
WARNA_MODEL = {MODEL_A: "#64748b", MODEL_B: "#3b82f6", MODEL_C: "#10b981"}

# Ruang pencarian hyperparameter (Model C)
PARAM_GRID = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
    "classifier__min_samples_split": [2, 20, 50, 100],
    "classifier__min_samples_leaf": [1, 5, 10, 20, 50, 100],
}

# Metrik cross-validation. Recall Fatal dihitung khusus untuk kelas Fatal injury.
SCORING = {
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "recall_fatal": make_scorer(recall_score, labels=[FATAL],
                                average="macro", zero_division=0),
}

# Aturan pemilihan model (DITETAPKAN SEBELUM melihat hasil data uji):
# skor gabungan = rata-rata CV balanced accuracy, CV macro F1, dan CV recall Fatal.
# Ketiganya adalah prioritas evaluasi proyek. Macro F1 ikut dihitung agar model
# tidak "asal menebak Fatal" (banyak alarm palsu menurunkan precision & F1).
SKOR_GABUNGAN = ["balanced_accuracy", "f1_macro", "recall_fatal"]


# =====================================================================
# 1. DATA
# =====================================================================
def load_data():
    """Membaca dataset dan mengambil 11 fitur + target saja."""
    df = pd.read_csv(DATA_PATH)
    return df[FEATURES + [TARGET]].copy()


def clean_data(df):
    """
    Merapikan penulisan kategori tanpa mengubah maknanya:
    - '?' yang rusak diganti '-'   : 'Lorry (41?100Q)' -> 'Lorry (41-100Q)'
    - 'unknown' disatukan dengan 'Unknown'
    NaN dibiarkan (ditangani SimpleImputer di pipeline).
    'Unknown' dibiarkan sebagai kategori tersendiri.
    """
    df = df.copy()
    for kolom in CATEGORICAL_FEATURES:
        df[kolom] = (df[kolom]
                     .str.strip()
                     .str.replace("?", "-", regex=False)
                     .replace("unknown", "Unknown"))
    return df


def split_data(df):
    """Membagi data latih dan uji dengan proporsi kelas yang sama (stratify)."""
    return train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=df[TARGET])


def make_cv():
    """Stratified K-Fold yang sama untuk semua model (perbandingan adil)."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# =====================================================================
# 2. PREPROCESSING DAN MODEL
# =====================================================================
def build_preprocessor():
    """ColumnTransformer: aturan preprocessing untuk tiap jenis kolom."""
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown='ignore': kategori tak dikenal saat prediksi tidak
        # membuat error (semua kolom one-hot fitur itu bernilai 0)
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("num", numeric_pipeline, NUMERIC_FEATURES),
    ])


def make_pipeline(**tree_params):
    """Pipeline lengkap: preprocessing + DecisionTreeClassifier."""
    params = {"class_weight": "balanced", "random_state": RANDOM_STATE}
    params.update(tree_params)
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(**params)),
    ])


def build_models():
    """Model pembanding dari tahap sebelumnya (A dan B)."""
    return {MODEL_A: make_pipeline(max_depth=None), MODEL_B: make_pipeline(max_depth=4)}


# =====================================================================
# 3. HYPERPARAMETER TUNING (MODEL C)
# =====================================================================
def tuning_model(X_train, y_train):
    """
    GridSearchCV pada data latih saja (data uji tidak disentuh).
    Mengembalikan parameter terbaik menurut skor gabungan dan tabel hasil lengkap.
    """
    n_kombinasi = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    print(f"\nGridSearchCV: {n_kombinasi} kombinasi x {CV_FOLDS} fold "
          f"= {n_kombinasi * CV_FOLDS} kali training. Mohon tunggu...")
    mulai = time.time()

    search = GridSearchCV(
        make_pipeline(), PARAM_GRID, scoring=SCORING, cv=make_cv(),
        refit=False,             # refit manual dengan skor gabungan di bawah
        return_train_score=True,  # untuk analisis overfitting (latih vs validasi)
        n_jobs=N_JOBS,
    )
    search.fit(X_train, y_train)
    print(f"Selesai dalam {time.time() - mulai:.0f} detik.")

    hasil = pd.DataFrame(search.cv_results_)
    hasil["skor_gabungan"] = hasil[[f"mean_test_{m}" for m in SKOR_GABUNGAN]].mean(axis=1)
    # Urutan: skor gabungan tertinggi; jika sama, pilih pohon yang lebih sederhana
    hasil["_depth_urut"] = hasil["param_classifier__max_depth"].fillna(999).astype(float)
    hasil = hasil.sort_values(
        ["skor_gabungan", "_depth_urut", "param_classifier__min_samples_leaf"],
        ascending=[False, True, False]).reset_index(drop=True)
    terbaik = hasil.loc[0, "params"]
    return terbaik, hasil, n_kombinasi


def ringkas_tuning(hasil, n=10):
    """Tabel ringkas kombinasi terbaik (untuk laporan & aplikasi)."""
    kolom = {
        "param_classifier__criterion": "criterion",
        "param_classifier__max_depth": "max_depth",
        "param_classifier__min_samples_split": "min_samples_split",
        "param_classifier__min_samples_leaf": "min_samples_leaf",
        "mean_test_balanced_accuracy": "cv_balanced_accuracy",
        "mean_test_f1_macro": "cv_f1_macro",
        "mean_test_recall_fatal": "cv_recall_fatal",
        "std_test_recall_fatal": "cv_recall_fatal_std",
        "mean_train_balanced_accuracy": "train_balanced_accuracy",
        "skor_gabungan": "skor_gabungan",
    }
    tabel = hasil[list(kolom)].rename(columns=kolom).copy()
    tabel["max_depth"] = tabel["max_depth"].apply(lambda v: "None" if pd.isna(v) else int(v))
    tabel.insert(0, "peringkat", range(1, len(tabel) + 1))
    return tabel.head(n) if n else tabel


def kurva_kedalaman(hasil, terbaik):
    """
    Skor latih vs validasi untuk setiap max_depth, dengan parameter lain
    sama seperti Model C. Diambil langsung dari hasil GridSearchCV (bukan training baru).
    """
    p = terbaik
    mask = ((hasil["param_classifier__criterion"] == p["classifier__criterion"])
            & (hasil["param_classifier__min_samples_split"] == p["classifier__min_samples_split"])
            & (hasil["param_classifier__min_samples_leaf"] == p["classifier__min_samples_leaf"]))
    data = hasil[mask].copy()
    data["depth"] = data["_depth_urut"]
    data = data.sort_values("depth")
    return [{
        "max_depth": "None" if d == 999 else int(d),
        "train_balanced_accuracy": r["mean_train_balanced_accuracy"],
        "cv_balanced_accuracy": r["mean_test_balanced_accuracy"],
        "train_f1_macro": r["mean_train_f1_macro"],
        "cv_f1_macro": r["mean_test_f1_macro"],
    } for d, (_, r) in zip(data["depth"], data.iterrows())]


# =====================================================================
# 4. EVALUASI
# =====================================================================
def hitung_metrik(y_true, y_pred):
    """Metrik utama. 'macro' = rata-rata setara untuk tiap kelas."""
    hasil = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    recall_kelas = recall_score(y_true, y_pred, labels=CLASS_ORDER,
                                average=None, zero_division=0)
    for kelas, nilai in zip(CLASS_ORDER, recall_kelas):
        hasil[f"recall_{kelas}"] = nilai
    return hasil


def evaluasi_model(nama, pipeline, X_train, X_test, y_train, y_test):
    """Cross-validation, training, lalu evaluasi pada data latih dan uji."""
    print("\n" + "=" * 70 + f"\n{nama}\n" + "=" * 70)
    params = pipeline.named_steps["classifier"].get_params()
    print("Parameter: " + ", ".join(f"{k}={params[k]}" for k in
          ["criterion", "max_depth", "min_samples_split", "min_samples_leaf"]))

    cv_hasil = cross_validate(pipeline, X_train, y_train, cv=make_cv(), scoring=SCORING)
    cv_ringkas = {}
    for metrik in SCORING:
        skor = cv_hasil[f"test_{metrik}"]
        cv_ringkas[metrik] = (skor.mean(), skor.std())
        print(f"CV {metrik:17s} ({CV_FOLDS}-fold, data latih): "
              f"{skor.mean():.4f} +/- {skor.std():.4f}")
    gabungan = np.mean([cv_ringkas[m][0] for m in SKOR_GABUNGAN])
    print(f"CV skor gabungan                       : {gabungan:.4f}")

    pipeline.fit(X_train, y_train)
    tree = pipeline.named_steps["classifier"]
    print(f"Kedalaman pohon : {tree.get_depth()}")
    print(f"Jumlah daun     : {tree.get_n_leaves()}")

    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)
    metrik_train = hitung_metrik(y_train, pred_train)
    metrik_test = hitung_metrik(y_test, pred_test)
    print(f"\nAccuracy          latih: {metrik_train['accuracy']:.4f} | uji: {metrik_test['accuracy']:.4f}")
    print(f"Balanced accuracy latih: {metrik_train['balanced_accuracy']:.4f} | uji: {metrik_test['balanced_accuracy']:.4f}")
    print(f"Macro F1          latih: {metrik_train['f1_macro']:.4f} | uji: {metrik_test['f1_macro']:.4f}")

    kwargs = dict(labels=CLASS_ORDER, digits=4, zero_division=0)
    laporan_test = classification_report(y_test, pred_test, **kwargs)
    cm = confusion_matrix(y_test, pred_test, labels=CLASS_ORDER)
    cm_train = confusion_matrix(y_train, pred_train, labels=CLASS_ORDER)
    print("\nClassification report (DATA UJI):\n" + laporan_test)
    print("Confusion matrix (DATA UJI) - baris = label asli, kolom = prediksi:")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_string())

    return {
        "nama": nama,
        "slug": SLUG[nama],
        "pipeline": pipeline,
        "params": {k: params[k] for k in
                   ["criterion", "max_depth", "min_samples_split", "min_samples_leaf", "class_weight"]},
        "cv": cv_ringkas,
        "cv_gabungan": gabungan,
        "depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "train": metrik_train,
        "test": metrik_test,
        "laporan_train": classification_report(y_train, pred_train, **kwargs),
        "laporan_test": laporan_test,
        "laporan_test_dict": classification_report(y_test, pred_test, output_dict=True, **kwargs),
        "cm": cm,
        "cm_train": cm_train,
    }


def buat_tabel_perbandingan(hasil_semua):
    """Tabel perbandingan semua model (CV, latih, uji, dan selisihnya)."""
    baris = []
    for h in hasil_semua:
        row = {"model": h["nama"], "kedalaman": h["depth"], "jumlah_daun": h["n_leaves"]}
        row.update({f"param_{k}": v for k, v in h["params"].items() if k != "class_weight"})
        for metrik, (rata, std) in h["cv"].items():
            row[f"cv_{metrik}_mean"] = rata
            row[f"cv_{metrik}_std"] = std
        row["cv_skor_gabungan"] = h["cv_gabungan"]
        for bagian in ["train", "test"]:
            for metrik, nilai in h[bagian].items():
                row[f"{bagian}_{metrik}"] = nilai
        # Selisih latih-uji: makin besar, makin kuat indikasi overfitting
        row["gap_balanced_accuracy"] = h["train"]["balanced_accuracy"] - h["test"]["balanced_accuracy"]
        row["gap_f1_macro"] = h["train"]["f1_macro"] - h["test"]["f1_macro"]
        baris.append(row)
    return pd.DataFrame(baris).round(4)


def pilih_model_terbaik(hasil_semua):
    """Model untuk aplikasi = CV skor gabungan tertinggi (data latih saja)."""
    return max(hasil_semua, key=lambda h: h["cv_gabungan"])


def status_overfitting(h):
    """Klasifikasi sederhana indikasi overfitting dari selisih latih-uji."""
    gap = h["train"]["balanced_accuracy"] - h["test"]["balanced_accuracy"]
    if gap > 0.15:
        return "kuat", gap
    if gap > 0.05:
        return "ringan", gap
    return "tidak terlihat", gap


# =====================================================================
# 5. NAMA FITUR HASIL ONE-HOT ENCODING
# =====================================================================
def daftar_fitur_encoded(pipeline):
    """
    Daftar (fitur_asli, kategori) untuk setiap kolom hasil preprocessing,
    sesuai urutan kolom yang dilihat pohon. Kolom numerik: kategori None.
    """
    pre = pipeline.named_steps["preprocessor"]
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    hasil = []
    for fitur, kategori_list in zip(CATEGORICAL_FEATURES, ohe.categories_):
        for kategori in kategori_list:
            hasil.append((fitur, str(kategori)))
    for fitur in NUMERIC_FEATURES:
        hasil.append((fitur, None))
    return hasil


def nama_fitur_tampil(fitur_encoded):
    """Nama yang mudah dibaca, misal 'Light_conditions = Daylight'."""
    return [f if k is None else f"{f} = {k}" for f, k in fitur_encoded]


# =====================================================================
# 6. VISUALISASI
# =====================================================================
def simpan_gambar(fig, nama_file, salin_ke_static=True):
    """Simpan ke reports/ dan (opsional) salin ke public/static/images/ untuk web."""
    path = REPORT_DIR / nama_file
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if salin_ke_static:
        shutil.copy(path, STATIC_IMG_DIR / nama_file)
    print(f"[tersimpan] reports/{nama_file}")


def plot_confusion_matrix(hasil_semua):
    """Heatmap confusion matrix (data uji) semua model berdampingan."""
    fig, axes = plt.subplots(1, len(hasil_semua), figsize=(7 * len(hasil_semua), 5.5))
    for ax, h in zip(axes, hasil_semua):
        sns.heatmap(h["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
        ax.set_title(f"{h['nama']}\nAccuracy uji = {h['test']['accuracy']:.2%}, "
                     f"Balanced acc = {h['test']['balanced_accuracy']:.3f}")
        ax.set_xlabel("Label Prediksi")
        ax.set_ylabel("Label Sebenarnya")
    plt.tight_layout()
    simpan_gambar(fig, "confusion_matrix.png")


def plot_perbandingan(hasil_semua):
    """Grafik batang perbandingan metrik semua model (data uji)."""
    metrik = {
        "Accuracy": "accuracy",
        "Balanced\naccuracy": "balanced_accuracy",
        "Macro\nprecision": "precision_macro",
        "Macro\nrecall": "recall_macro",
        "Macro F1": "f1_macro",
        "Recall\nSlight": "recall_Slight Injury",
        "Recall\nSerious": "recall_Serious Injury",
        "Recall\nFatal": "recall_Fatal injury",
    }
    data = pd.DataFrame({h["nama"]: [h["test"][m] for m in metrik.values()]
                         for h in hasil_semua}, index=list(metrik.keys()))
    ax = data.plot(kind="bar", figsize=(14, 5.5), width=0.8,
                   color=[WARNA_MODEL[h["nama"]] for h in hasil_semua])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor (data uji)")
    ax.set_title("Perbandingan Performa Model pada Data Uji")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper right")
    plt.tight_layout()
    simpan_gambar(ax.figure, "model_comparison.png")


def plot_latih_vs_uji(hasil_semua):
    """Balanced accuracy & macro F1 di data latih vs data uji (indikasi overfitting)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metrik, judul in zip(axes, ["balanced_accuracy", "f1_macro"],
                                 ["Balanced Accuracy", "Macro F1"]):
        nama = [h["nama"].split(" (")[0] for h in hasil_semua]
        latih = [h["train"][metrik] for h in hasil_semua]
        uji = [h["test"][metrik] for h in hasil_semua]
        x = np.arange(len(nama))
        b1 = ax.bar(x - 0.2, latih, 0.4, label="Data latih", color="#94a3b8")
        b2 = ax.bar(x + 0.2, uji, 0.4, label="Data uji", color="#10b981")
        ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
        ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(x, nama)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{judul}: Latih vs Uji")
        ax.legend(loc="upper right")
    plt.tight_layout()
    simpan_gambar(fig, "train_vs_test.png")


def plot_kurva_kedalaman(kurva, terbaik):
    """Validation curve dari GridSearchCV: skor latih vs validasi per max_depth."""
    label = [str(k["max_depth"]) for k in kurva]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(label, [k["train_balanced_accuracy"] for k in kurva], "o-", color="#94a3b8",
            label="Balanced acc. - latih (rata-rata fold)")
    ax.plot(label, [k["cv_balanced_accuracy"] for k in kurva], "o-", color="#10b981",
            label="Balanced acc. - validasi CV")
    ax.plot(label, [k["cv_f1_macro"] for k in kurva], "s--", color="#3b82f6",
            label="Macro F1 - validasi CV")
    terpilih = str(terbaik["classifier__max_depth"])
    if terpilih in label:
        ax.axvline(label.index(terpilih), color="#ef4444", linestyle=":", label=f"max_depth Model C = {terpilih}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    ax.set_title("Kurva Kedalaman Pohon (parameter lain = Model C)\n"
                 "Jarak latih-validasi yang melebar = tanda overfitting")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    simpan_gambar(fig, "tuning_depth_curve.png")


def plot_pohon(pipeline, nama_tampil, nama_model, nama_file, batas_tampil=4):
    """
    Visualisasi pohon dengan nama fitur hasil one-hot yang jelas.
    Pohon yang lebih dalam dari batas_tampil hanya digambar sampai level tersebut
    agar tetap terbaca (aturan lengkap ada di decision_rules_*.txt).
    """
    tree = pipeline.named_steps["classifier"]
    depth = tree.get_depth()
    terpotong = depth > batas_tampil
    lebar = 48 if not terpotong else 56
    fig, ax = plt.subplots(figsize=(lebar, 14))
    plot_tree(tree, feature_names=nama_tampil, class_names=list(tree.classes_),
              filled=True, rounded=True, impurity=True, proportion=False, precision=1,
              fontsize=10, ax=ax, max_depth=batas_tampil if terpotong else None)
    catatan = (f" - ditampilkan {batas_tampil} level pertama dari {depth} level"
               if terpotong else "")
    ax.set_title(f"Pohon Keputusan {nama_model}{catatan}\n"
                 "Kondisi 'fitur = kategori <= 0.5' bernilai True (ke kiri) jika kategori "
                 "tersebut TIDAK terpenuhi. 'value' = bobot kelas setelah class_weight, "
                 f"urutan {list(tree.classes_)}", fontsize=18)
    simpan_gambar(fig, nama_file)
    return terpotong


def hitung_importance(pipeline, fitur_encoded):
    """Feature importance per kolom one-hot dan per fitur asli (dijumlahkan)."""
    tree = pipeline.named_steps["classifier"]
    per_kolom = pd.DataFrame({
        "fitur": [f for f, _ in fitur_encoded],
        "nama": nama_fitur_tampil(fitur_encoded),
        "importance": tree.feature_importances_,
    })
    per_fitur = (per_kolom.groupby("fitur")["importance"].sum()
                 .reindex(FEATURES).fillna(0).sort_values(ascending=False))
    per_kolom = per_kolom[per_kolom["importance"] > 0].sort_values("importance", ascending=False)
    return per_kolom, per_fitur


def plot_importance(per_kolom, per_fitur, nama_model, nama_file, maks_kategori=15):
    """Dua grafik: per fitur asli (11 fitur) dan per kategori hasil one-hot."""
    per_kolom = per_kolom.head(maks_kategori)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    sns.barplot(x=per_fitur.values, y=per_fitur.index, color="#3b82f6", ax=axes[0])
    axes[0].set_title(f"Feature Importance per Fitur Asli - {nama_model}\n"
                      "(jumlah importance semua kategorinya)")
    axes[0].set_xlabel("Importance")
    axes[0].set_ylabel("")
    sns.barplot(x="importance", y="nama", data=per_kolom, color="#10b981", ax=axes[1])
    axes[1].set_title(f"{maks_kategori} Kategori (One-Hot) Terpenting\n(importance > 0)")
    axes[1].set_xlabel("Importance")
    axes[1].set_ylabel("")
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    simpan_gambar(fig, nama_file)


# =====================================================================
# 7. ATURAN IF-THEN DARI POHON
# =====================================================================
def kondisi_ke_teks(fitur, kategori, threshold, ke_kanan):
    """Menerjemahkan satu kondisi split menjadi kalimat yang mudah dibaca."""
    if kategori is None:  # fitur numerik (jumlah kendaraan, bilangan bulat)
        batas = int(np.floor(threshold))
        return f"{fitur} >= {batas + 1}" if ke_kanan else f"{fitur} <= {batas}"
    return f"{fitur} = {kategori}" if ke_kanan else f"{fitur} BUKAN {kategori}"


def rapikan_kondisi(kondisi):
    """
    Menggabungkan kondisi pada fitur yang sama, misalnya:
    'X BUKAN a' + 'X BUKAN b'  -> 'X bukan (a, b)'
    'X = a' + 'X BUKAN b'      -> 'X = a'
    'N <= 3' + 'N >= 2'        -> '2 <= N <= 3'
    """
    per_fitur, urutan = {}, []
    for fitur, teks, jenis, nilai in kondisi:
        if fitur not in per_fitur:
            per_fitur[fitur] = {"sama": None, "bukan": [], "min": None, "max": None}
            urutan.append(fitur)
        info = per_fitur[fitur]
        if jenis == "sama":
            info["sama"] = nilai
        elif jenis == "bukan":
            info["bukan"].append(nilai)
        elif jenis == "min":
            info["min"] = nilai if info["min"] is None else max(info["min"], nilai)
        else:
            info["max"] = nilai if info["max"] is None else min(info["max"], nilai)

    hasil = []
    for fitur in urutan:
        info = per_fitur[fitur]
        if info["min"] is not None or info["max"] is not None:
            if info["min"] is not None and info["max"] is not None:
                hasil.append(f"{info['min']} <= {fitur} <= {info['max']}")
            elif info["min"] is not None:
                hasil.append(f"{fitur} >= {info['min']}")
            else:
                hasil.append(f"{fitur} <= {info['max']}")
        elif info["sama"] is not None:
            hasil.append(f"{fitur} = {info['sama']}")
        elif len(info["bukan"]) == 1:
            hasil.append(f"{fitur} bukan {info['bukan'][0]}")
        else:
            hasil.append(f"{fitur} bukan ({', '.join(info['bukan'])})")
    return hasil


def ekstrak_aturan(pipeline, fitur_encoded, X_train, y_train):
    """
    Menelusuri semua jalur dari akar ke daun.
    Statistik tiap daun dihitung dari jumlah data latih ASLI yang jatuh ke daun
    tersebut (bukan nilai berbobot class_weight), agar mudah dipahami.
    """
    tree = pipeline.named_steps["classifier"]
    t = tree.tree_
    X_encoded = pipeline.named_steps["preprocessor"].transform(X_train)
    leaf_id = tree.apply(X_encoded)
    y_arr = np.asarray(y_train)
    aturan = []

    def telusuri(node, kondisi):
        if t.children_left[node] == t.children_right[node]:  # daun
            mask = leaf_id == node
            jumlah = {k: int((y_arr[mask] == k).sum()) for k in CLASS_ORDER}
            total = int(mask.sum())
            aturan.append({
                "leaf": int(node),
                "kondisi": rapikan_kondisi(kondisi),
                "prediksi": str(tree.classes_[int(np.argmax(t.value[node][0]))]),
                "jumlah_data": total,
                "jumlah_per_kelas": jumlah,
                "persen_per_kelas": {k: (v / total * 100 if total else 0) for k, v in jumlah.items()},
            })
            return
        fitur, kategori = fitur_encoded[t.feature[node]]
        thr = t.threshold[node]
        for ke_kanan, anak in [(False, t.children_left[node]), (True, t.children_right[node])]:
            teks = kondisi_ke_teks(fitur, kategori, thr, ke_kanan)
            if kategori is None:
                batas = int(np.floor(thr))
                jenis, nilai = ("min", batas + 1) if ke_kanan else ("max", batas)
            else:
                jenis, nilai = ("sama" if ke_kanan else "bukan"), kategori
            telusuri(anak, kondisi + [(fitur, teks, jenis, nilai)])

    telusuri(0, [])
    return aturan


def jelaskan_aturan(rule, persen_global):
    """Penjelasan aturan dalam bahasa Indonesia yang mudah dipahami."""
    p = rule["persen_per_kelas"]
    kalimat = (f"Dari {rule['jumlah_data']} kecelakaan di data latih yang memenuhi kondisi ini, "
               f"{p['Slight Injury']:.1f}% luka ringan, {p['Serious Injury']:.1f}% luka serius, "
               f"dan {p['Fatal injury']:.1f}% fatal.")
    kelas = rule["prediksi"]
    if kelas != "Slight Injury":
        rasio = p[kelas] / persen_global[kelas] if persen_global[kelas] else 0
        kalimat += (f" Persentase {kelas} di kelompok ini {rasio:.1f} kali lipat dari rata-rata "
                    f"seluruh data ({persen_global[kelas]:.2f}%). Karena class_weight='balanced' "
                    f"memberi bobot lebih besar pada kelas langka, model memprediksi {kelas} "
                    f"walaupun mayoritas kecelakaan di kelompok ini tetap luka ringan. "
                    f"Artinya, prediksi ini lebih tepat dibaca sebagai 'risiko {kelas} lebih tinggi'.")
    else:
        kalimat += " Kelompok ini didominasi luka ringan, sehingga model memprediksi Slight Injury."
    return kalimat


def pilih_contoh_aturan(aturan, persen_global):
    """Ambil 1 aturan per kelas prediksi: daun dengan data latih terbanyak."""
    contoh = []
    for kelas in CLASS_ORDER:
        kandidat = [r for r in aturan if r["prediksi"] == kelas]
        if kandidat:
            terbaik = dict(max(kandidat, key=lambda r: r["jumlah_data"]))
            terbaik["penjelasan"] = jelaskan_aturan(terbaik, persen_global)
            contoh.append(terbaik)
    return contoh


def format_aturan(rule, nomor=None):
    kepala = f"Aturan {nomor}" if nomor else f"Daun {rule['leaf']}"
    baris = [f"{kepala} (prediksi: {rule['prediksi']}, n={rule['jumlah_data']})", "IF"]
    for i, k in enumerate(rule["kondisi"]):
        baris.append(f"   {'AND ' if i else '    '}{k}")
    baris.append(f"THEN {rule['prediksi']}")
    return "\n".join(baris)


def simpan_aturan(aturan, contoh, nama_model, nama_file):
    with open(REPORT_DIR / nama_file, "w", encoding="utf-8") as f:
        f.write(f"ATURAN IF-THEN {nama_model}\n")
        f.write("Statistik = jumlah data latih asli yang jatuh ke daun tersebut.\n\n")
        f.write("=" * 70 + "\nCONTOH ATURAN (1 per kelas prediksi)\n" + "=" * 70 + "\n\n")
        for i, r in enumerate(contoh, start=1):
            f.write(format_aturan(r, i) + "\n")
            f.write("Penjelasan: " + r["penjelasan"] + "\n\n")
        f.write("=" * 70 + f"\nSEMUA ATURAN ({len(aturan)} daun)\n" + "=" * 70 + "\n\n")
        for r in sorted(aturan, key=lambda r: -r["jumlah_data"]):
            f.write(format_aturan(r) + "\n")
            f.write("Distribusi: " + ", ".join(
                f"{k} {v} ({r['persen_per_kelas'][k]:.1f}%)"
                for k, v in r["jumlah_per_kelas"].items()) + "\n\n")
    print(f"[tersimpan] reports/{nama_file}")


def interpretasi_model(h, X_train, y_train, persen_global):
    """Pohon, feature importance, dan aturan IF-THEN untuk satu model."""
    pipeline, nama, slug = h["pipeline"], h["nama"], h["slug"]
    fitur_encoded = daftar_fitur_encoded(pipeline)
    file_pohon = f"decision_tree_{slug}.png"
    file_imp = f"feature_importance_{slug}.png"
    terpotong = plot_pohon(pipeline, nama_fitur_tampil(fitur_encoded), nama, file_pohon)
    per_kolom, per_fitur = hitung_importance(pipeline, fitur_encoded)
    plot_importance(per_kolom, per_fitur, nama.split(" (")[0], file_imp)
    aturan = ekstrak_aturan(pipeline, fitur_encoded, X_train, y_train)
    contoh = pilih_contoh_aturan(aturan, persen_global)
    simpan_aturan(aturan, contoh, nama, f"decision_rules_{slug}.txt")

    print(f"\nFeature importance {nama} (per fitur asli):")
    print(per_fitur.round(4).to_string())
    print("\n" + "-" * 70 + f"\nCONTOH ATURAN IF-THEN ({nama})\n" + "-" * 70)
    for i, r in enumerate(contoh, start=1):
        print("\n" + format_aturan(r, i))
        print("Penjelasan: " + r["penjelasan"])

    return {
        "nama": nama,
        "slug": slug,
        "kedalaman": h["depth"],
        "jumlah_daun": h["n_leaves"],
        "jumlah_kolom_onehot": len(fitur_encoded),
        "gambar_pohon": file_pohon,
        "gambar_importance": file_imp,
        "pohon_terpotong": terpotong,
        "feature_importance": per_fitur.to_dict(),
        "importance_kategori": per_kolom.head(15)[["nama", "importance"]].to_dict("records"),
        "contoh_aturan": contoh,
        "semua_aturan": sorted(aturan, key=lambda r: -r["jumlah_data"]),
    }


# =====================================================================
# 8. SIMPAN LAPORAN, MODEL, DAN METADATA
# =====================================================================
def simpan_laporan(tabel, hasil_semua, terbaik, tuning_tabel):
    tabel.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    tuning_tabel.to_csv(REPORT_DIR / "tuning_results.csv", index=False)
    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT - MODEL A, B, dan C\n")
        f.write(f"Data uji = {int(TEST_SIZE * 100)}% (stratify, random_state={RANDOM_STATE})\n")
        f.write(f"Model terpilih (CV skor gabungan {' + '.join(SKOR_GABUNGAN)}): {terbaik['nama']}\n\n")
        for h in hasil_semua:
            f.write("=" * 70 + f"\n{h['nama']}\n" + "=" * 70 + "\n")
            f.write(f"Parameter: {h['params']}\n")
            f.write(f"Kedalaman: {h['depth']} | Jumlah daun: {h['n_leaves']}\n")
            for metrik, (rata, std) in h["cv"].items():
                f.write(f"CV {metrik}: {rata:.4f} +/- {std:.4f}\n")
            f.write(f"CV skor gabungan: {h['cv_gabungan']:.4f}\n")
            f.write("\n--- DATA LATIH ---\n" + h["laporan_train"] + "\n")
            f.write("--- DATA UJI ---\n" + h["laporan_test"] + "\n")
    print("[tersimpan] reports/model_comparison.csv")
    print("[tersimpan] reports/tuning_results.csv")
    print("[tersimpan] reports/classification_report.txt")


def ke_python(obj):
    """Ubah tipe numpy menjadi tipe Python biasa agar bisa disimpan ke JSON."""
    if isinstance(obj, dict):
        return {str(k): ke_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ke_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return ke_python(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else round(float(obj), 4)
    return obj


def simpan_model_dan_metadata(df, X_train, terbaik, hasil_semua, interpretasi,
                              persen_global, tuning):
    """
    best_model.joblib  : pipeline lengkap (preprocessing + Decision Tree terpilih).
    model_metadata.json: info untuk aplikasi Flask. Semua berasal dari hasil nyata.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = terbaik["pipeline"]
    joblib.dump(pipeline, MODEL_DIR / "best_model.joblib")
    print("[tersimpan] models/best_model.joblib")

    ohe = (pipeline.named_steps["preprocessor"]
           .named_transformers_["cat"].named_steps["onehot"])
    pilihan = {f: [str(c) for c in cats] for f, cats in zip(CATEGORICAL_FEATURES, ohe.categories_)}
    rentang_numerik = {f: {"min": int(X_train[f].min()), "max": int(X_train[f].max())}
                       for f in NUMERIC_FEATURES}

    metadata = {
        "model_terpilih": terbaik["nama"],
        "slug_terpilih": terbaik["slug"],
        "kriteria_pemilihan": ("CV skor gabungan tertinggi = rata-rata CV balanced accuracy, "
                               f"macro F1, dan recall Fatal ({CV_FOLDS}-fold, data latih)"),
        "versi_sklearn": sklearn.__version__,
        "fitur_kategorikal": CATEGORICAL_FEATURES,
        "fitur_numerik": NUMERIC_FEATURES,
        "urutan_kelas": CLASS_ORDER,
        "pilihan_kategori": pilihan,
        "rentang_numerik": rentang_numerik,
        "dataset": {
            "jumlah_baris": len(df),
            "jumlah_fitur": len(FEATURES),
            "jumlah_kelas": df[TARGET].nunique(),
            "distribusi_kelas": df[TARGET].value_counts().reindex(CLASS_ORDER).to_dict(),
            "jumlah_latih": len(X_train),
            "jumlah_uji": len(df) - len(X_train),
        },
        "baseline": tuning["baseline"],
        "evaluasi": [{
            "nama": h["nama"],
            "slug": h["slug"],
            "params": h["params"],
            "kedalaman": h["depth"],
            "jumlah_daun": h["n_leaves"],
            "cv": {m: {"mean": r, "std": s} for m, (r, s) in h["cv"].items()},
            "cv_gabungan": h["cv_gabungan"],
            "overfitting": dict(zip(["status", "gap_balanced_accuracy"], status_overfitting(h))),
            "train": h["train"],
            "test": h["test"],
            "classification_report_test": h["laporan_test_dict"],
            "confusion_matrix_test": h["cm"].tolist(),
            "confusion_matrix_train": h["cm_train"].tolist(),
        } for h in hasil_semua],
        "tuning": {
            "param_grid": {k.replace("classifier__", ""): [("None" if v is None else v) for v in vals]
                           for k, vals in PARAM_GRID.items()},
            "jumlah_kombinasi": tuning["n_kombinasi"],
            "cv_folds": CV_FOLDS,
            "parameter_terbaik": {k.replace("classifier__", ""): v
                                  for k, v in tuning["terbaik"].items()},
            "top": ringkas_tuning(tuning["hasil"], 10).to_dict("records"),
            "kurva_kedalaman": tuning["kurva"],
        },
        "interpretasi": interpretasi,        # per slug model
        "persen_global_latih": persen_global,
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ke_python(metadata), f, ensure_ascii=False, indent=2)
    print("[tersimpan] models/model_metadata.json")


def uji_muat_model(X_test):
    """Memastikan model yang disimpan bisa dimuat ulang dan dipakai memprediksi."""
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    contoh = X_test.head(3)
    pred = model.predict(contoh)
    proba = model.predict_proba(contoh)
    print("\nUji muat ulang best_model.joblib (3 data uji pertama):")
    for i, (p, pr) in enumerate(zip(pred, proba), start=1):
        rincian = ", ".join(f"{k}: {v:.2f}" for k, v in zip(model.classes_, pr))
        print(f"  Data {i}: {p}  |  {rincian}")
    aneh = X_test.head(1).copy()
    aneh["Type_of_vehicle"] = "Kategori Tidak Dikenal"
    aneh["Driving_experience"] = np.nan
    print(f"  Input dengan kategori asing + NaN -> {model.predict(aneh)[0]} (tidak error)")


# =====================================================================
# PROGRAM UTAMA
# =====================================================================
def main():
    for folder in [REPORT_DIR, MODEL_DIR, STATIC_IMG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Data latih: {len(X_train)} baris | Data uji: {len(X_test)} baris")
    print("Distribusi kelas data uji:")
    print(y_test.value_counts().reindex(CLASS_ORDER).to_string())

    baseline = hitung_metrik(y_test, ["Slight Injury"] * len(y_test))
    print(f"\nBaseline 'selalu Slight Injury' -> accuracy uji {baseline['accuracy']:.4f}, "
          f"macro F1 {baseline['f1_macro']:.4f}, "
          f"balanced accuracy {baseline['balanced_accuracy']:.4f}")

    # --- Tuning Model C (data latih saja) ---
    terbaik_params, hasil_tuning, n_kombinasi = tuning_model(X_train, y_train)
    print("\n10 kombinasi terbaik (CV, data latih):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(ringkas_tuning(hasil_tuning, 10).round(4).to_string(index=False))
    print(f"\nParameter Model C: {terbaik_params}")
    kurva = kurva_kedalaman(hasil_tuning, terbaik_params)

    # --- Training & evaluasi tiga model ---
    modelku = build_models()
    modelku[MODEL_C] = make_pipeline(**{k.replace("classifier__", ""): v
                                        for k, v in terbaik_params.items()})
    hasil_semua = [evaluasi_model(nama, pipe, X_train, X_test, y_train, y_test)
                   for nama, pipe in modelku.items()]
    tabel = buat_tabel_perbandingan(hasil_semua)
    terbaik = pilih_model_terbaik(hasil_semua)

    print("\n" + "=" * 70 + "\nPERBANDINGAN MODEL (ringkas)\n" + "=" * 70)
    kolom = ["model", "kedalaman", "jumlah_daun",
             "cv_balanced_accuracy_mean", "cv_f1_macro_mean", "cv_recall_fatal_mean", "cv_skor_gabungan",
             "train_balanced_accuracy", "test_balanced_accuracy", "train_f1_macro", "test_f1_macro",
             "test_accuracy", "test_precision_macro", "test_recall_macro",
             "test_recall_Slight Injury", "test_recall_Serious Injury", "test_recall_Fatal injury",
             "gap_balanced_accuracy"]
    with pd.option_context("display.width", 200):
        print(tabel[kolom].set_index("model").T.to_string())
    for h in hasil_semua:
        status, gap = status_overfitting(h)
        print(f"Indikasi overfitting {h['nama']}: {status} (selisih balanced acc. latih-uji = {gap:.3f})")
    print(f"\nModel terpilih (CV skor gabungan): {terbaik['nama']}")

    simpan_laporan(tabel, hasil_semua, terbaik, ringkas_tuning(hasil_tuning, None))
    plot_confusion_matrix(hasil_semua)
    plot_perbandingan(hasil_semua)
    plot_latih_vs_uji(hasil_semua)
    plot_kurva_kedalaman(kurva, terbaik_params)

    # --- Interpretasi: Model B (syarat TA-05) dan model terpilih ---
    persen_global = (y_train.value_counts(normalize=True) * 100).to_dict()
    perlu = [h for h in hasil_semua if h["nama"] == MODEL_B]
    if terbaik["nama"] != MODEL_B:
        perlu.append(terbaik)
    interpretasi = {}
    for h in perlu:
        print("\n" + "=" * 70 + f"\nINTERPRETASI {h['nama']}\n" + "=" * 70)
        interpretasi[h["slug"]] = interpretasi_model(h, X_train, y_train, persen_global)
    # Salinan aturan model terpilih dengan nama lama (kompatibel dengan laporan sebelumnya)
    shutil.copy(REPORT_DIR / f"decision_rules_{terbaik['slug']}.txt",
                REPORT_DIR / "decision_rules.txt")

    # --- Simpan model terpilih ---
    print()
    simpan_model_dan_metadata(df, X_train, terbaik, hasil_semua, interpretasi, persen_global, {
        "baseline": baseline, "terbaik": terbaik_params, "hasil": hasil_tuning,
        "n_kombinasi": n_kombinasi, "kurva": kurva,
    })
    uji_muat_model(X_test)


if __name__ == "__main__":
    main()
