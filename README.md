# Smartphone AI Advisor

Aplikasi rekomendasi dan perbandingan smartphone berbasis FastAPI (backend) + HTML static (frontend).

---

## Struktur Folder

```
App_Smartphone/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── dataset_katalog.csv
│   ├── dataset_vektor.csv
│   └── scaler_knn.pkl
└── frontend/
    └── index.html
```

---

## Deploy ke Vessel

### 1. Backend (Python / FastAPI)

1. Buat **New Service** → pilih **Python**
2. Set **Root Directory** ke `backend/`
3. **Build Command** : `pip install -r requirements.txt`
4. **Start Command** : `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Setelah deploy, copy **Public URL** backend (contoh: `https://smartphone-api.vessel.app`)

### 2. Frontend (Static Site)

1. Buat **New Service** → pilih **Static Site**
2. Set **Root Directory** ke `frontend/`
3. Tambahkan  **Environment Variable** :
   * Key: `BACKEND_URL`
   * Value: URL backend dari langkah 1 (contoh: `https://smartphone-api.vessel.app`)
4. Sebelum deploy, edit `index.html` baris pertama JavaScript:
   ```js
   const API = 'https://smartphone-api.vessel.app';  // ganti dengan URL backend Anda
   ```

---

## Jalankan Lokal

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
# Buka frontend/index.html langsung di browser, atau:
cd frontend
python -m http.server 3000
# Akses: http://localhost:3000
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

## Catatan Penting

* `dataset_katalog.csv`, `dataset_vektor.csv`, dan `scaler_knn.pkl` **wajib ada** di folder `backend/` — tidak boleh dihapus.
* Backend menggunakan CORS `allow_origins=["*"]` — aman untuk development. Untuk production, ganti dengan domain frontend spesifik.
