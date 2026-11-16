// script.js - interaksi kecil di sisi browser (validasi tetap dilakukan di server)
document.addEventListener("DOMContentLoaded", function () {
  // 1) Buka/tutup sidebar di layar kecil
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebarBackdrop");
  const toggle = document.getElementById("sidebarToggle");
  function setSidebar(open) {
    sidebar.classList.toggle("open", open);
    backdrop.classList.toggle("show", open);
  }
  if (toggle) toggle.addEventListener("click", () => setSidebar(!sidebar.classList.contains("open")));
  if (backdrop) backdrop.addEventListener("click", () => setSidebar(false));

  const form = document.getElementById("predictForm");
  if (!form) return;

  // 2) Validasi ringan sebelum dikirim: tandai isian yang masih kosong
  form.addEventListener("submit", function (e) {
    let kosong = null;
    form.querySelectorAll("select, input").forEach(function (el) {
      const isEmpty = !el.value || el.value.trim() === "";
      el.classList.toggle("is-invalid", isEmpty);
      if (isEmpty && !kosong) kosong = el;
    });
    if (kosong) {
      e.preventDefault();
      kosong.focus();
      kosong.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    // Tampilkan status "memproses" pada tombol
    const btn = document.getElementById("btnPredict");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Memproses...';
  });

  // Hilangkan tanda merah saat pengguna mulai mengisi
  form.querySelectorAll("select, input").forEach(function (el) {
    el.addEventListener("change", () => el.classList.remove("is-invalid"));
  });

  // 3) Tombol "Isi contoh": mengisi form dengan satu skenario agar mudah dicoba
  const contoh = {
    Age_band_of_driver: "18-30",
    Sex_of_driver: "Male",
    Driving_experience: "2-5yr",
    Type_of_vehicle: "Automobile",
    Number_of_vehicles_involved: "1",
    Type_of_collision: "Collision with pedestrians",
    Road_surface_conditions: "Dry",
    Road_surface_type: "Asphalt roads",
    Weather_conditions: "Normal",
    Light_conditions: "Darkness - lights lit",
    Types_of_Junction: "No junction",
  };
  const btnContoh = document.getElementById("btnContoh");
  if (btnContoh) {
    btnContoh.addEventListener("click", function () {
      Object.entries(contoh).forEach(([nama, nilai]) => {
        const el = form.elements[nama];
        if (!el) return;
        // Hanya isi jika nilai tersebut memang ada di pilihan dropdown
        if (el.tagName === "SELECT" && ![...el.options].some(o => o.value === nilai)) return;
        el.value = nilai;
        el.classList.remove("is-invalid");
      });
    });
  }

  // 4) Setelah prediksi, gulir ke kartu hasil (berguna di layar kecil)
  const hasil = document.getElementById("hasil");
  if (hasil && window.innerWidth < 1200) hasil.scrollIntoView({ behavior: "smooth" });
});

// 5) Filter daftar aturan di halaman Visualisasi (teks & kelas prediksi)
document.addEventListener("DOMContentLoaded", function () {
  const teks = document.getElementById("ruleFilter");
  const kelas = document.getElementById("ruleClass");
  const items = document.querySelectorAll("#ruleList .rule-item");
  const kosong = document.getElementById("ruleEmpty");
  if (!teks || !kelas || !items.length) return;

  function terapkan() {
    const q = teks.value.trim().toLowerCase();
    const k = kelas.value;
    let tampil = 0;
    items.forEach(function (el) {
      const cocok = (!q || el.dataset.text.includes(q)) && (!k || el.dataset.class === k);
      el.hidden = !cocok;
      if (cocok) tampil++;
    });
    kosong.hidden = tampil > 0;
  }
  teks.addEventListener("input", terapkan);
  kelas.addEventListener("change", terapkan);
});

// 6) Gambar pohon: geser otomatis ke tengah agar simpul akar langsung terlihat
window.addEventListener("load", function () {
  document.querySelectorAll(".tree-scroll").forEach(function (box) {
    box.scrollLeft = (box.scrollWidth - box.clientWidth) / 2;
  });
});
