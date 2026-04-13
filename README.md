# 📱 Smartphone AI Advisor

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80_Live_Demo-smart--phn--syqd.vercel.app-4361ee?style=for-the-badge)](https://smart-phn-syqd.vercel.app/)
[![Backend API](https://img.shields.io/badge/%E2%9A%99%EF%B8%8F_Backend_API-smart--phn.vercel.app-00c896?style=for-the-badge)](https://smart-phn.vercel.app/api/ping)

Aplikasi rekomendasi dan perbandingan smartphone berbasis **FastAPI** (backend) + **HTML static** (frontend), di-deploy terpisah di Vercel.

> 🌐 **Demo langsung:** [https://smart-phn-syqd.vercel.app](https://smart-phn-syqd.vercel.app/)

---

## Struktur Folder

```
App_Smartphone/
├── backend/
│   ├── api/
│   │   └── index.py           ← Entry point Vercel (Mangum adapter)
│   ├── main.py                ← FastAPI app (API engine)
│   ├── requirements.txt       ← Library Python
│   ├── vercel.json            ← Konfigurasi routing Vercel
│   ├── dataset_katalog.csv    ← Data HP (395 item)
│   ├── dataset_vektor.csv     ← Data fitur ML
│   └── scaler_knn.pkl         ← Model scaler (MinMaxScaler)
└── frontend/
    └── index.html             ← UI aplikasi (static)
```

---

## Evaluasi Model

Hasil evaluasi dijalankan terhadap **90 skenario pengujian** menggunakan `cekmodel.py` dengan metrik standar Information Retrieval.

```
==================================================================
 🏆 LAPORAN EVALUASI MACHINE LEARNING (KONTEN-BERBASIS IR) 🏆
==================================================================
Total Skenario Diuji   : 90 Skenario
Skenario 'Masuk Akal'  : 74 Skenario (Ada Ground Truth di DB)
Waktu Pengujian        : 0.58 detik
------------------------------------------------------------------
 📊 1. METRIK INFORMATION RETRIEVAL (STRICT METRICS)
       (Hanya dihitung pada skenario masuk akal)
       - Precision@5 : 57.03%  (Berapa % rekomendasi yang tepat sasaran?)
       - Recall@5    : 10.79%  (Berapa % HP relevan di DB yg berhasil ditarik?)
       - MRR         : 52.66%  (Seberapa cepat jawaban benar muncul di urutan atas?)
       - NDCG@5      : 52.73%  (Kualitas urutan ranking, bobot tinggi di rank 1)
------------------------------------------------------------------
 🎯 2. FLEKSIBILITAS MODEL (HEURISTIC PROPORTIONAL)
       - Akurasi     : 93.98%  (Kemampuan AI bernegosiasi dgn user)
==================================================================
```

**Interpretasi singkat:**

| Metrik            | Nilai  | Catatan                                                                         |
| ----------------- | ------ | ------------------------------------------------------------------------------- |
| Precision@5       | 57.03% | Lebih dari separuh rekomendasi yang muncul memang relevan dengan kebutuhan user |
| Recall@5          | 10.79% | Wajar rendah — katalog besar (395 HP), sistem hanya menampilkan 3–5 teratas   |
| MRR               | 52.66% | Rata-rata jawaban paling relevan muncul di posisi ke-2 hasil teratas            |
| NDCG@5            | 52.73% | Kualitas urutan ranking cukup baik; item terbaik cenderung muncul di rank 1     |
| Akurasi Heuristik | 93.98% | Model mampu mengakomodasi variasi input user secara fleksibel                   |

> Recall yang rendah bukan indikasi masalah — ini karakteristik umum sistem rekomendasi dengan kandidat besar dan output terbatas (Top-3). Precision dan NDCG yang di atas 50% menunjukkan model bekerja dengan baik untuk use case ini.

---

## URL Deployment (Live)

| Service          | URL                                   |
| ---------------- | ------------------------------------- |
| 🌐 Frontend      | https://smart-phn-syqd.vercel.app     |
| ⚙️ Backend API | https://smart-phn.vercel.app          |
| 🔍 Health Check  | https://smart-phn.vercel.app/api/ping |

---

## Deploy ke Vercel (Step-by-Step)

> Backend dan frontend di-deploy sebagai **dua project Vercel terpisah** dari satu repo GitHub yang sama.

---

### LANGKAH 1 — Push ke GitHub

Buka terminal di folder `App_Smartphone`, jalankan satu per satu:

```bash
git init
git add .
git commit -m "init: Smartphone AI Advisor"
```

Buka [github.com/new](https://github.com/new), buat repo baru:

* Nama bebas, contoh: `smartphone-ai-advisor`
* Visibility: Public atau Private
* **Jangan** centang "Add a README file"
* Klik **Create repository**

Kembali ke terminal:

```bash
git remote add origin https://github.com/USERNAME/smartphone-ai-advisor.git
git branch -M main
git push -u origin main
```

Pastikan semua file terupload termasuk `dataset_katalog.csv`, `dataset_vektor.csv`, dan `scaler_knn.pkl`.

---

### LANGKAH 2 — Deploy Backend di Vercel

1. Buka [vercel.com](https://vercel.com/) → Login dengan akun GitHub
2. Klik **Add New Project**
3. Pilih repo dari daftar → Klik **Import**
4. Pada halaman  **Configure Project** , isi:

   | Field                      | Nilai                                                           |
   | -------------------------- | --------------------------------------------------------------- |
   | **Project Name**     | bebas (contoh:`smart-phn`)                                    |
   | **Framework Preset** | `Other`                                                       |
   | **Root Directory**   | Klik**Edit**→ ketik `backend`→ klik**Continue** |
   | **Build Command**    | *(kosongkan)*                                                 |
   | **Output Directory** | *(kosongkan)*                                                 |
   | **Install Command**  | *(kosongkan)*                                                 |
5. Klik **Deploy** — tunggu sampai muncul tanda ✅ **Congratulations!**
6. Klik **Continue to Dashboard** → copy **Domain** yang tampil.
7. Verifikasi backend berjalan — buka di browser:

   ```
   https://NAMA-PROJECT.vercel.app/api/ping
   ```

   Harus menampilkan:

   ```json
   {"status": "ok", "total_hp": 395}
   ```

> ⚠️ **Penting:** Pastikan folder `backend/api/index.py` ada di repo sebelum deploy. File ini adalah entry point yang wajib ada agar Vercel mendeteksi Python function.

---

### LANGKAH 3 — Update URL Backend di Frontend

Buka file `frontend/index.html`, cari baris berikut (sekitar baris 746):

```js
const API = 'GANTI_DENGAN_URL_BACKEND_VERCEL'; // ← Ganti ini dengan URL backend Vercel Anda
```

Ganti dengan URL dari Langkah 2:

```js
const API = 'https://smartphone-ai-backend.vercel.app';
```

Simpan file, lalu push ke GitHub:

```bash
git add frontend/index.html
git commit -m "fix: set backend URL to Vercel"
git push
```

---

### LANGKAH 4 — Deploy Frontend di Vercel

1. Kembali ke [vercel.com/new](https://vercel.com/new)
2. Klik **Add New Project** → Import repo yang **sama** (`smartphone-ai-advisor`)
3. Pada halaman  **Configure Project** , isi:| Field                      | Nilai                                                            |
   | -------------------------- | ---------------------------------------------------------------- |
   | **Project Name**     | `smartphone-ai-frontend`(bebas)                                |
   | **Framework Preset** | `Other`                                                        |
   | **Root Directory**   | Klik**Edit**→ ketik `frontend`→ klik**Continue** |
   | **Build Command**    | *(kosongkan)*                                                  |
   | **Output Directory** | *(kosongkan)*                                                  |
4. Klik **Deploy** → tunggu ✅
5. URL frontend siap diakses dan dibagikan, contoh:
   ```
   https://smartphone-ai-frontend.vercel.app
   ```

---

### LANGKAH 5 — Verifikasi Akhir

Buka URL frontend di browser. Lakukan tes berikut:

* [ ] Slider budget, RAM, ROM bisa digerakkan
* [ ] Ketik query di chat input, tekan Enter → hasil muncul
* [ ] Klik **Eksekusi Pencarian** → 3 kartu HP muncul dengan radar chart
* [ ] Buka tab **Perbandingan HP** → pilih 2 HP → klik **Mulai Perbandingan**
* [ ] Semua skor, tabel, dan kesimpulan muncul dengan benar

---

## Jalankan Lokal (Development)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# Akses API: http://127.0.0.1:8000

# Frontend (terminal baru)
cd frontend
python -m http.server 3000
# Akses UI: http://localhost:3000
```

Saat lokal, pastikan baris di `index.html` adalah:

```js
const API = 'http://127.0.0.1:8000';
```

---

## Catatan Penting

* `dataset_katalog.csv`, `dataset_vektor.csv`, dan `scaler_knn.pkl` **wajib ada** di folder `backend/` dan ikut di-push ke GitHub — ketiga file ini dibaca langsung saat startup API.
* Backend di Vercel berjalan sebagai **serverless function** — request pertama setelah idle (cold start) bisa lebih lambat 2–3 detik. Request berikutnya normal.
* Backend menggunakan CORS `allow_origins=["*"]` — cukup untuk production skala kecil. Untuk keamanan lebih ketat, ganti `"*"` dengan domain frontend spesifik di `main.py`.
