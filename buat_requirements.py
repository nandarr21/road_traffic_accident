from importlib import metadata
from pathlib import Path

# Library yang dipakai app.py saat aplikasi berjalan
RUNTIME = ["flask", "pandas", "numpy", "scikit-learn", "scipy", "joblib"]
# Server produksi untuk Render (Linux). Tidak dipakai Vercel, tetapi ukurannya kecil.
TAMBAHAN = ["gunicorn>=22.0"]

baris = ["# DIBUAT OTOMATIS oleh buat_requirements.py - jangan diedit manual.",
         "# Versi sama persis dengan laptop tempat model dilatih.",
         "# Untuk menyiapkan lingkungan training/EDA, gunakan requirements-dev.txt."]
for paket in RUNTIME:
    try:
        baris.append(f"{paket}=={metadata.version(paket)}")
    except metadata.PackageNotFoundError:
        raise SystemExit(f"Library '{paket}' belum terpasang. Jalankan: pip install -r requirements-dev.txt")
baris += TAMBAHAN

path = Path(__file__).resolve().parent / "requirements.txt"
path.write_text("\n".join(baris) + "\n", encoding="utf-8")
print(f"requirements.txt dibuat ({path}):")
print("\n".join(baris[3:]))
