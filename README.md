# RTA Classifier: Klasifikasi Tingkat Keparahan Kecelakaan Lalu Lintas (Decision Tree)

Tugas Akhir Praktikum **TA-05 Pembelajaran Mesin**.

Aplikasi web Flask yang mengklasifikasikan tingkat keparahan kecelakaan lalu lintas
(**Slight Injury**, **Serious Injury**, **Fatal injury**) berdasarkan faktor pengemudi,
kendaraan, dan kondisi lingkungan, menggunakan algoritma **Decision Tree** (Scikit-Learn).

- **Dataset:** Road Traffic Accident Dataset (`RTA Dataset.csv`), Addis Ababa, Ethiopia, 2017–2020. Isinya 12.316 baris dan 32 kolom.
- **Target (y):** `Accident_severity`.
- **Fitur (X), 11 kolom:** `Age_band_of_driver`, `Sex_of_driver`, `Driving_experience`,
  `Type_of_vehicle`, `Road_surface_conditions`, `Weather_conditions`, `Light_conditions`,
  `Type_of_collision`, `Road_surface_type`, `Types_of_Junction`, `Number_of_vehicles_involved`.

---

## 1. Struktur proyek

```
road_traffic_accident/
├── app.py                 # Aplikasi Flask (memuat model, tidak melakukan training)
├── train_model.py         # Preprocessing, tuning, evaluasi, visualisasi, simpan model
├── eda.py                 # Exploratory Data Analysis (grafik ke reports/eda/)
├── check_dataset.py       # Pemeriksaan awal dataset
├── RTA Dataset.csv
├── requirements-dev.txt   # Library lengkap untuk training & EDA (lokal)
├── requirements.txt       # Library aplikasi, versi terkunci (dibuat buat_requirements.py)
├── buat_requirements.py   # Membuat requirements.txt dari versi di laptop
├── .vercelignore, .python-version            # Konfigurasi deployment Vercel
├── Procfile, render.yaml                     # Konfigurasi deployment Render (alternatif)
├── models/
│   ├── best_model.joblib       # Pipeline lengkap (preprocessing + Decision Tree terpilih)
│   └── model_metadata.json     # Metrik, pilihan dropdown, aturan IF-THEN, hasil tuning
├── reports/               # Laporan hasil eksperimen (CSV, TXT, PNG)
├── public/static/         # CSS, JS, dan gambar grafik (disajikan CDN di Vercel)
└── templates/             # base, index, predict, evaluation, visualization, error
```

## 2. Cara menjalankan di Windows (VS Code PowerShell)

```powershell
cd road_traffic_accident
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python check_dataset.py   # opsional: cek dataset
python eda.py             # opsional: grafik EDA
python train_model.py     # wajib: training + tuning (sekitar 2–6 menit)
python buat_requirements.py  # wajib sebelum deploy: kunci versi library
python app.py             # jalankan aplikasi
```

Setelah itu buka **http://127.0.0.1:5000** di browser.

- Kalau `Activate.ps1` diblokir, jalankan dulu: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Kalau laptop terasa berat saat tuning, batasi jumlah inti CPU yang dipakai:
  `$env:N_JOBS=2; python train_model.py`.
- Bootstrap, ikon, dan font dimuat dari CDN, jadi browser perlu terhubung ke internet.
- `train_model.py` wajib dijalankan ulang setiap kali versi scikit-learn berubah.
  Aplikasi akan menampilkan peringatan kalau versinya tidak cocok.

## 3. Metodologi

1. **Pembersihan teks.** Tanda `?` yang rusak di nama kategori diganti `-` (contoh: `Lorry (41?100Q)` menjadi `Lorry (41-100Q)`), dan `unknown` disatukan dengan `Unknown`. Langkah ini tidak menghitung statistik apa pun dari data.
2. **Pembagian data.** `train_test_split(test_size=0.2, random_state=42, stratify=y)` menghasilkan 9.852 data latih dan 2.464 data uji.
3. **Preprocessing** dengan `Pipeline` dan `ColumnTransformer`:
   - kategorikal: `SimpleImputer(most_frequent)` lalu `OneHotEncoder(handle_unknown='ignore')`
   - numerik: `SimpleImputer(median)`
4. **Tiga model** dengan preprocessing dan pembagian data yang sama. Semua memakai `class_weight='balanced'` dan `random_state=42`:
   - **Model A (Unpruned):** `max_depth=None`
   - **Model B (Pruned):** `max_depth=4`
   - **Model C (Tuned):** hasil `GridSearchCV` dengan **Stratified 5-Fold CV** atas
     `criterion` {gini, entropy}, `max_depth` {3, 4, 5, 6, 8, 10, 12, None},
     `min_samples_split` {2, 20, 50, 100}, dan `min_samples_leaf` {1, 5, 10, 20, 50, 100}.
     Totalnya 384 kombinasi.
5. **Pemilihan model untuk aplikasi.** Model dengan **skor gabungan CV** tertinggi dipilih. Skor gabungan adalah rata-rata dari CV balanced accuracy, CV macro F1, dan CV recall Fatal.
6. **Pencegahan data leakage.** Data uji tidak dipakai saat tuning maupun pemilihan model, dan hanya dipakai sekali di akhir untuk evaluasi. Imputer dan encoder berada di dalam pipeline, sehingga di setiap fold CV keduanya hanya dipelajari dari bagian latih fold tersebut.

## 4. Hasil (data uji, 2.464 baris: 2.084 Slight, 349 Serious, 31 Fatal)

| Metrik (data uji)     | Baseline* | Model A | Model B | **Model C** |
|-----------------------|-----------|---------|---------|-------------|
| Accuracy              | 0.846     | 0.694   | 0.597   | 0.561       |
| Balanced accuracy     | 0.333     | 0.365   | 0.423   | **0.436**   |
| Macro precision       | 0.282     | 0.358   | 0.388   | 0.370       |
| Macro recall          | 0.333     | 0.365   | 0.423   | 0.436       |
| Macro F1              | 0.306     | 0.356   | 0.338   | 0.340       |
| Recall Fatal          | 0.000     | 0.032   | 0.419   | 0.355       |
| CV skor gabungan      | –         | 0.272   | 0.377   | **0.417**   |
| Balanced acc. latih   | –         | 0.936   | 0.488   | 0.548       |

\*Baseline = selalu menebak "Slight Injury".

**Model C** (`criterion=entropy`, `max_depth=8`, `min_samples_split=100`,
`min_samples_leaf=20`; 67 daun) dipakai oleh aplikasi.

**Interpretasi jujur:**

- **Model A overfitting kuat.** Balanced accuracy-nya 0,936 di data latih tetapi hanya 0,365 di data uji. Pohonnya punya 40 level dan 2.702 daun.
- **Hasil tuning hanya sedikit lebih baik.** Model C unggul dalam balanced accuracy dan recall Serious (0,355 vs 0,180). Tapi recall Fatal-nya di data uji **lebih rendah** dari Model B (11 vs 13 dari 31 kasus), dan accuracy-nya paling rendah. Selisih 2 kasus fatal ini masih dalam batas variasi: standar deviasi CV recall Fatal sekitar 0,10–0,13.
- **Semua model hanya sedikit di atas baseline** untuk macro F1. Ke-11 fitur ini memiliki daya prediksi yang terbatas terhadap tingkat keparahan kecelakaan.
- **Accuracy yang tinggi menyesatkan untuk data ini.** Baseline yang selalu menebak "Slight Injury" sudah mencapai accuracy 84,6% tanpa mendeteksi satu pun kasus serius atau fatal.
- **Prediksi Fatal sebaiknya dibaca sebagai "risiko lebih tinggi".** Karena `class_weight='balanced'`, model memprediksi Fatal untuk kelompok yang persentase fatalnya beberapa kali di atas rata-rata (1,29%), walaupun mayoritas kasus di kelompok itu tetap luka ringan. Untuk alasan yang sama, nilai `predict_proba` adalah skor berbobot, bukan peluang nyata.

Angka lengkap tersedia di `reports/model_comparison.csv`, `reports/tuning_results.csv`,
dan `reports/classification_report.txt`.

## 5. Fitur aplikasi

| Halaman | Isi |
|---|---|
| **Dashboard** | Ringkasan dataset, metrik model aktif dibanding baseline, parameter hasil tuning, perbandingan 3 model, distribusi target, 5 faktor terpenting |
| **Prediksi** | Form 11 variabel dengan dropdown dari kategori data latih; probabilitas ketiga kelas; persentase kasus nyata dari daun pohon; jalur IF-THEN yang dilalui input |
| **Evaluasi Model** | Tabel latih vs uji untuk 3 model dan baseline, CV 5-fold, indikasi overfitting, classification report, confusion matrix, top-10 hasil tuning, kurva kedalaman, grafik perbandingan |
| **Visualisasi** | Pohon keputusan (bisa memilih Model B atau Model C), feature importance, 3 contoh aturan beserta penjelasan, daftar semua aturan dengan pencarian dan filter kelas |

**Validasi input:** isian wajib diisi; kategori harus ada di daftar data latih; jumlah
kendaraan harus bilangan bulat 1–7; input kosong, huruf, desimal, NaN, ∞, dan teks yang
terlalu panjang ditolak dengan pesan yang jelas. Halaman error 404, 405, dan 500 sudah
ditangani, begitu juga kondisi saat model belum ada atau metadata berasal dari versi lama.

## 6. Deployment ke Vercel

**Persiapan (di laptop, sekali saja setiap kali model dilatih ulang):**

```powershell
python train_model.py
python buat_requirements.py
```

`buat_requirements.py` menulis `requirements.txt` berisi versi flask, pandas, numpy,
scikit-learn, scipy, dan joblib yang **sama persis** dengan laptop Anda. Ini wajib,
karena `best_model.joblib` hanya aman dimuat dengan versi scikit-learn yang sama
dengan saat training. matplotlib dan seaborn tidak ikut, agar ukuran deployment kecil.

**Hal yang sudah disesuaikan untuk Vercel:**
- `app.py` berisi objek Flask bernama `app` dan dikenali otomatis oleh Vercel (tanpa `vercel.json`).
- File statis berada di `public/static/`, sehingga Vercel menyajikannya lewat CDN
  dengan alamat `/static/...` yang sama seperti saat lokal.
- `.python-version` berisi `3.12`, karena Vercel mendukung Python 3.12–3.14.
- `.vercelignore` mencegah `venv/`, dataset, `reports/`, dan script training ikut di-upload.

**Cara A: lewat GitHub (disarankan)**
1. Push proyek ke repositori GitHub. Folder `models/`, `public/`, dan `requirements.txt` wajib ikut.
2. Buka https://vercel.com/new, pilih **Import** pada repositori tersebut.
3. Framework Preset akan terdeteksi sebagai **Flask**. Biarkan pengaturan lain apa adanya, lalu klik **Deploy**.
4. Buka `https://<nama-proyek>.vercel.app/health`. Responsnya harus `{"status": "ok", ...}`.

**Cara B: lewat Vercel CLI** (perlu Node.js)

```powershell
npm i -g vercel
vercel login
vercel          # deploy preview
vercel --prod   # deploy production
```

Setiap kali model dilatih ulang, jalankan lagi `python buat_requirements.py`, lalu
commit dan push (atau `vercel --prod`).

## 7. Deployment ke Render (alternatif)

1. Jalankan `python train_model.py` dan `python buat_requirements.py` secara lokal,
   lalu **commit folder `models/`, `public/`, dan file `requirements.txt`**. Render tidak
   menjalankan training.
2. Push proyek ke GitHub.
3. Di Render: **New → Web Service**, lalu pilih repositori.
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - Environment variable: `PYTHON_VERSION=3.12.7`
   - (Alternatif: gunakan **Blueprint** dengan file `render.yaml`.)
4. Setelah deploy, cek `https://<nama-app>.onrender.com/health`. Responsnya harus `{"status": "ok", ...}`.

Port dibaca dari environment variable `PORT` (`app.py` dan `Procfile`).
