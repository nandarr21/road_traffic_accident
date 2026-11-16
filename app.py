import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, render_template, request

# ---------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
# Kunci metadata yang wajib ada (format train_model.py versi terbaru)
METADATA_WAJIB = ["model_terpilih", "slug_terpilih", "evaluasi", "interpretasi",
                  "tuning", "pilihan_kategori", "rentang_numerik", "dataset"]

# File statis (CSS, JS, gambar) berada di public/static agar:
# - di Vercel disajikan langsung oleh CDN dari folder public/ (alamat /static/...)
# - saat lokal tetap disajikan Flask dengan alamat yang sama (/static/...)
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "public" / "static"), static_url_path="/static")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rta")

# Label tampilan (bahasa Indonesia) untuk setiap fitur; nilai kategori tetap
# memakai teks asli dataset karena itulah yang dikenali model.
LABEL_FITUR = {
    "Age_band_of_driver": ("Kelompok usia pengemudi", "Rentang usia pengemudi"),
    "Sex_of_driver": ("Jenis kelamin pengemudi", "Jenis kelamin pengemudi"),
    "Driving_experience": ("Pengalaman mengemudi", "Lama pengalaman mengemudi"),
    "Type_of_vehicle": ("Jenis kendaraan", "Kendaraan yang dikemudikan"),
    "Road_surface_conditions": ("Kondisi permukaan jalan", "Kering, basah, dll."),
    "Weather_conditions": ("Kondisi cuaca", "Cuaca saat kecelakaan"),
    "Light_conditions": ("Kondisi pencahayaan", "Siang atau gelap"),
    "Type_of_collision": ("Jenis tabrakan", "Bentuk tabrakan yang terjadi"),
    "Road_surface_type": ("Jenis permukaan jalan", "Aspal, tanah, kerikil, dll."),
    "Types_of_Junction": ("Jenis persimpangan", "Bentuk persimpangan lokasi"),
    "Number_of_vehicles_involved": ("Jumlah kendaraan terlibat", "Bilangan bulat"),
}

# Kelompok fitur di form agar tidak terlalu padat
KELOMPOK_FORM = [
    ("Faktor Pengemudi", "bi-person",
     ["Age_band_of_driver", "Sex_of_driver", "Driving_experience"]),
    ("Faktor Kendaraan", "bi-truck",
     ["Type_of_vehicle", "Number_of_vehicles_involved", "Type_of_collision"]),
    ("Kondisi Lingkungan", "bi-cloud-sun",
     ["Road_surface_conditions", "Road_surface_type", "Weather_conditions",
      "Light_conditions", "Types_of_Junction"]),
]

INFO_KELAS = {
    "Slight Injury": {
        "label": "Luka Ringan", "warna": "slight", "ikon": "bi-shield-check",
        "penjelasan": "Kondisi yang dimasukkan mirip dengan kelompok kecelakaan yang "
                      "umumnya hanya menimbulkan luka ringan.",
    },
    "Serious Injury": {
        "label": "Luka Serius", "warna": "serious", "ikon": "bi-exclamation-triangle",
        "penjelasan": "Kondisi yang dimasukkan termasuk kelompok dengan persentase luka "
                      "serius lebih tinggi daripada rata-rata data.",
    },
    "Fatal injury": {
        "label": "Fatal", "warna": "fatal", "ikon": "bi-exclamation-octagon",
        "penjelasan": "Kondisi yang dimasukkan termasuk kelompok dengan persentase "
                      "kecelakaan fatal lebih tinggi daripada rata-rata data.",
    },
}

WARNA_MODEL = {"model_a": "slate", "model_b": "blue", "model_c": "green"}

# Batas panjang input teks (mencegah input berlebihan)
MAKS_PANJANG_INPUT = 100


# ---------------------------------------------------------------------
# Memuat model dan metadata (sekali saat aplikasi dijalankan)
# ---------------------------------------------------------------------
def load_artifacts():
    """Memuat model dan metadata. Mengembalikan (model, meta, error, peringatan)."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None, ("Model belum tersedia. Jalankan 'python train_model.py' "
                            "terlebih dahulu untuk membuat folder models/."), None
    try:
        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, None, f"File model_metadata.json rusak atau tidak terbaca: {e}", None

    hilang = [k for k in METADATA_WAJIB if k not in metadata]
    if hilang:
        return None, None, ("Format model_metadata.json berasal dari versi lama "
                            f"(tidak ada: {', '.join(hilang)}). Jalankan ulang "
                            "'python train_model.py'."), None

    peringatan = None
    versi_latih = metadata.get("versi_sklearn")
    if versi_latih and versi_latih != sklearn.__version__:
        peringatan = (f"Model dilatih dengan scikit-learn {versi_latih}, sedangkan yang "
                      f"terpasang {sklearn.__version__}. Jika hasil terasa aneh, jalankan "
                      "ulang 'python train_model.py'.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # peringatan versi sudah ditangani di atas
            model = joblib.load(MODEL_PATH)
    except Exception as e:  # misalnya file rusak / versi scikit-learn tidak kompatibel
        return None, None, (f"Gagal memuat best_model.joblib ({type(e).__name__}: {e}). "
                            "Jalankan ulang 'python train_model.py'."), None
    return model, metadata, None, peringatan


# Pastikan folder pendukung ada (misalnya jika proyek disalin tidak lengkap)
for _folder in ["templates", "public/static/css", "public/static/js"]:
    if not (BASE_DIR / _folder).is_dir():
        log.error("Folder '%s' tidak ditemukan di %s. Salin ulang folder proyek secara lengkap.",
                  _folder, BASE_DIR)

MODEL, META, LOAD_ERROR, LOAD_WARNING = load_artifacts()
if LOAD_ERROR:
    log.error(LOAD_ERROR)
elif LOAD_WARNING:
    log.warning(LOAD_WARNING)


def semua_fitur():
    return META["fitur_kategorikal"] + META["fitur_numerik"]


def evaluasi_per_slug():
    return {e["slug"]: e for e in META["evaluasi"]}


def interpretasi_terpilih():
    return META["interpretasi"].get(META["slug_terpilih"], {})


# ---------------------------------------------------------------------
# Validasi input
# ---------------------------------------------------------------------
def validasi_input(form):
    """
    Memeriksa setiap isian form. Mengembalikan (data_bersih, daftar_error).
    - Kategorikal : wajib diisi dan harus salah satu kategori dari data latih.
    - Numerik     : wajib bilangan bulat dalam rentang data latih.
    """
    data, errors = {}, {}

    for fitur in META["fitur_kategorikal"]:
        nilai = (form.get(fitur) or "").strip()
        if not nilai:
            errors[fitur] = "Wajib dipilih."
        elif len(nilai) > MAKS_PANJANG_INPUT or nilai not in META["pilihan_kategori"][fitur]:
            errors[fitur] = "Pilihan tidak dikenal. Pilih dari daftar."
        else:
            data[fitur] = nilai

    for fitur in META["fitur_numerik"]:
        teks = (form.get(fitur) or "").strip().replace(",", ".")
        batas = META["rentang_numerik"][fitur]
        if not teks:
            errors[fitur] = "Wajib diisi."
            continue
        if len(teks) > 10:
            errors[fitur] = "Angka terlalu panjang."
            continue
        try:
            angka = float(teks)
        except ValueError:
            errors[fitur] = "Harus berupa angka."
            continue
        if angka != angka or angka in (float("inf"), float("-inf")):  # NaN / tak hingga
            errors[fitur] = "Harus berupa angka."
        elif not angka.is_integer():
            errors[fitur] = "Harus bilangan bulat."
        elif not batas["min"] <= angka <= batas["max"]:
            errors[fitur] = f"Harus antara {batas['min']} dan {batas['max']} (rentang data latih)."
        else:
            data[fitur] = int(angka)

    return data, errors


# ---------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------
def cari_aturan_daun(X_input):
    """Mencari aturan IF-THEN (daun pohon) yang dilalui input ini."""
    tree = MODEL.named_steps["classifier"]
    X_enc = MODEL.named_steps["preprocessor"].transform(X_input)
    leaf = int(tree.apply(X_enc)[0])
    for aturan in interpretasi_terpilih().get("semua_aturan", []):
        if aturan["leaf"] == leaf:
            return aturan
    log.warning("Daun %s tidak ditemukan di metadata; jalankan ulang train_model.py.", leaf)
    return None


def prediksi(data):
    """Menjalankan pipeline pada satu baris input dan menyusun hasil untuk template."""
    X_input = pd.DataFrame([data], columns=semua_fitur())
    kelas = str(MODEL.predict(X_input)[0])

    # Probabilitas ketiga kelas (predict_proba). Karena class_weight='balanced',
    # nilai ini sudah diberi bobot sehingga BUKAN peluang nyata di dunia sebenarnya.
    proba = MODEL.predict_proba(X_input)[0]
    peta = dict(zip(MODEL.classes_, proba))
    skor = [{"kelas": k, "label": INFO_KELAS[k]["label"], "warna": INFO_KELAS[k]["warna"],
             "nilai": float(peta.get(k, 0.0))} for k in META["urutan_kelas"]]

    aturan = cari_aturan_daun(X_input)
    persen_global = META.get("persen_global_latih", {})
    rasio = None
    if aturan and kelas != "Slight Injury" and persen_global.get(kelas):
        rasio = aturan["persen_per_kelas"][kelas] / persen_global[kelas]

    return {
        "kelas": kelas,
        "info": INFO_KELAS[kelas],
        "skor": skor,
        "aturan": aturan,
        "rasio": rasio,
        "persen_global": persen_global,
        "input": data,
    }


# ---------------------------------------------------------------------
# Data bantu untuk template
# ---------------------------------------------------------------------
@app.context_processor
def inject_globals():
    """Variabel yang tersedia di semua template."""
    return {
        "load_error": LOAD_ERROR,
        "load_warning": LOAD_WARNING,
        "info_kelas": INFO_KELAS,
        "label_fitur": LABEL_FITUR,
        "warna_model": WARNA_MODEL,
        "model_terpilih": META["model_terpilih"] if META else None,
        "slug_terpilih": META["slug_terpilih"] if META else None,
    }


@app.template_filter("persen")
def format_persen(nilai, desimal=1):
    """Contoh: 0.4229 -> '42.3%'"""
    try:
        return f"{float(nilai) * 100:.{desimal}f}%"
    except (TypeError, ValueError):
        return "-"


@app.template_filter("angka")
def format_angka(nilai, desimal=4):
    try:
        return f"{float(nilai):.{desimal}f}"
    except (TypeError, ValueError):
        return "-"


@app.template_filter("ribuan")
def format_ribuan(nilai):
    """Contoh: 12316 -> '12.316'"""
    try:
        return f"{int(nilai):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "-"


@app.template_filter("pendek")
def nama_pendek(nama):
    """'Model C (Tuned, GridSearchCV)' -> 'Model C'"""
    return str(nama).split(" (")[0]


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    if LOAD_ERROR:
        return render_template("index.html", page="dashboard")
    ds = META["dataset"]
    total = sum(ds["distribusi_kelas"].values())
    distribusi = [{"kelas": k, "label": INFO_KELAS[k]["label"], "warna": INFO_KELAS[k]["warna"],
                   "jumlah": ds["distribusi_kelas"][k], "persen": ds["distribusi_kelas"][k] / total}
                  for k in META["urutan_kelas"]]
    terpilih = evaluasi_per_slug()[META["slug_terpilih"]]
    return render_template("index.html", page="dashboard", ds=ds, distribusi=distribusi,
                           terpilih=terpilih, evaluasi=META["evaluasi"],
                           baseline=META.get("baseline", {}), tuning=META["tuning"],
                           importance=sorted(interpretasi_terpilih().get("feature_importance", {}).items(),
                                             key=lambda x: -x[1])[:5])


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if LOAD_ERROR:
        return render_template("predict.html", page="predict")

    hasil, errors, nilai_form = None, {}, {}
    if request.method == "POST":
        nilai_form = {k: v for k, v in request.form.items() if k in LABEL_FITUR}
        data, errors = validasi_input(request.form)
        if not errors:
            try:
                hasil = prediksi(data)
            except Exception:  # pengaman terakhir agar halaman tidak crash
                log.exception("Prediksi gagal")
                errors["_umum"] = ("Terjadi kesalahan saat memproses prediksi. Coba lagi; "
                                   "jika berulang, jalankan ulang 'python train_model.py'.")

    terpilih = evaluasi_per_slug()[META["slug_terpilih"]]
    return render_template("predict.html", page="predict", kelompok=KELOMPOK_FORM,
                           pilihan=META["pilihan_kategori"], rentang=META["rentang_numerik"],
                           fitur_numerik=META["fitur_numerik"], nilai=nilai_form,
                           errors=errors, hasil=hasil, terpilih=terpilih)


@app.route("/evaluation")
def evaluation():
    if LOAD_ERROR:
        return render_template("evaluation.html", page="evaluation")
    return render_template("evaluation.html", page="evaluation",
                           evaluasi=META["evaluasi"], kriteria=META["kriteria_pemilihan"],
                           urutan_kelas=META["urutan_kelas"], tuning=META["tuning"],
                           baseline=META.get("baseline", {}))


@app.route("/visualization")
def visualization():
    if LOAD_ERROR:
        return render_template("visualization.html", page="visualization")
    tersedia = META["interpretasi"]
    slug = request.args.get("model", META["slug_terpilih"])
    if slug not in tersedia:  # parameter tidak valid -> kembali ke model terpilih
        slug = META["slug_terpilih"]
    interp = tersedia[slug]
    importance = sorted(interp["feature_importance"].items(), key=lambda x: -x[1])
    maks = max((v for _, v in importance), default=0) or 1
    return render_template("visualization.html", page="visualization", interp=interp,
                           pilihan_model=[(s, tersedia[s]["nama"]) for s in tersedia],
                           slug_aktif=slug, importance=importance, maks=maks)


@app.route("/health")
def health():
    """Dipakai untuk mengecek aplikasi hidup (berguna saat deployment)."""
    return {"status": "ok" if not LOAD_ERROR else "model_missing",
            "model": META["model_terpilih"] if META else None}


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", page=None, kode=404,
                           pesan="Alamat yang Anda buka tidak tersedia."), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return render_template("error.html", page=None, kode=405,
                           pesan="Metode permintaan tidak diizinkan untuk alamat ini."), 405


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", page=None, kode=500,
                           pesan="Terjadi kesalahan di server. Detail tercatat di terminal."), 500


if __name__ == "__main__":
    # PORT dari environment variable (Render), default 5000 untuk lokal.
    # Mode debug aktif secara default saat dijalankan lokal dengan "python app.py".
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
