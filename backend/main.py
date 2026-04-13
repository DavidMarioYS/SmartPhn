from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
import os
from typing import List, Optional

app = FastAPI(title="Smartphone AI Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD DATA
# ==========================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
df_katalog = pd.read_csv(os.path.join(BASE_DIR, 'dataset_katalog.csv'))
df_vektor  = pd.read_csv(os.path.join(BASE_DIR, 'dataset_vektor.csv'))
scaler     = joblib.load(os.path.join(BASE_DIR, 'scaler_knn.pkl'))

df_katalog['Harga_Num'] = df_katalog['Harga_Varian'].astype(int)

# ─────────────────────────────────────────
# FIX #1 — Kolom Umur_HP tidak ada di CSV.
# Dihitung dari kolom Rilis (sama seperti saat dataset dibuat):
#   2026 -> 0, 2025 -> 1, 2024 -> 2, 2023 -> 3, 2022 -> 4, 2021 -> 5
# "Maret 2026" dan format non-angka diekstrak tahunnya dulu.
# ─────────────────────────────────────────
def _parse_tahun(rilis_val):
    m = re.search(r'(\d{4})', str(rilis_val))
    return int(m.group(1)) if m else 2025

BASE_YEAR = 2026
df_katalog['Umur_HP'] = df_katalog['Rilis'].apply(
    lambda r: max(0, BASE_YEAR - _parse_tahun(r))
)

# ─────────────────────────────────────────
# FIX #2 — Kolom Layar_Jenis tidak ada di CSV.
# Diekstrak dari nama chipset menggunakan heuristik:
#   Snapdragon 8xx / Dimensity 9xxx / Apple   -> AMOLED/OLED
#   Snapdragon 7xx / Dimensity 7xxx-8xxx       -> AMOLED (mid-range)
#   Snapdragon 6xx / Dimensity 6xxx / Helio   -> IPS LCD
#   Samsung selalu AMOLED di semua lini
# ─────────────────────────────────────────
def _infer_layar_jenis(chipset_str, brand_str=""):
    cs = str(chipset_str).upper()
    if str(brand_str).upper() == 'APPLE':
        return 'OLED'
    if re.search(r'SNAPDRAGON\s*8', cs):
        return 'AMOLED'
    if re.search(r'DIMENSITY\s*9\d{3}', cs):
        return 'AMOLED'
    if re.search(r'DIMENSITY\s*8\d{3}', cs):
        return 'AMOLED'
    if str(brand_str).upper() == 'SAMSUNG':
        return 'AMOLED'
    if re.search(r'DIMENSITY\s*7\d{3}', cs):
        return 'AMOLED'
    if re.search(r'SNAPDRAGON\s*7', cs):
        return 'AMOLED'
    return 'IPS LCD'

df_katalog['Layar_Jenis'] = df_katalog.apply(
    lambda r: _infer_layar_jenis(r['Chipset'], r['Brand']), axis=1
)

# ─────────────────────────────────────────
# FIX #3 — Inverse-transform dari scaler untuk mendapatkan
# nilai original Chipset_Skor dan Refresh dari df_vektor
# yang sudah dalam skala [0,1].
# ─────────────────────────────────────────
_COLS = list(df_vektor.columns)

def _inv(col, scaled_val):
    i  = _COLS.index(col)
    mn = float(scaler.data_min_[i])
    mx = float(scaler.data_max_[i])
    return scaled_val * (mx - mn) + mn

df_vektor['_Chip_orig']    = df_vektor['Chipset_Skor'].apply(lambda v: _inv('Chipset_Skor', v))
df_vektor['_Refresh_orig'] = df_vektor['Refresh'].apply(lambda v: round(_inv('Refresh', v)))

# ==========================================
# HELPER & SCORING FUNCTIONS
# (Identik dengan app_offline.py)
# ==========================================
def _ekstrak_mp(kamera_str):
    m = re.search(r'(\d+)\s*MP', str(kamera_str), re.IGNORECASE)
    return int(m.group(1)) if m else 12

def _skor_performa(ram, chip_orig):
    ram_map = {4: 45, 6: 60, 8: 72, 12: 85, 16: 93, 24: 100}
    ram_s   = min(100, ram_map.get(int(ram), 65))
    chip_s  = min(100, int(float(chip_orig) / 3 * 100))
    return int(ram_s * 0.45 + chip_s * 0.55)

def _skor_layar(layar_jenis, refresh):
    jenis = str(layar_jenis).upper()
    if   'LTPO'   in jenis:                    panel_s = 96
    elif 'AMOLED' in jenis or 'OLED' in jenis: panel_s = 88
    elif 'IPS'    in jenis:                    panel_s = 67
    elif 'TFT'    in jenis:                    panel_s = 50
    elif 'TN'     in jenis:                    panel_s = 35
    else:                                       panel_s = 60
    ref_map = {60: 50, 90: 68, 120: 82, 144: 93, 165: 96, 185: 100}
    ref_s   = ref_map.get(int(refresh), min(100, int(int(refresh) * 0.54)))
    return int(panel_s * 0.62 + ref_s * 0.38)

def _skor_baterai(baterai, charger):
    mah_s = min(100, max(0, int((int(baterai) - 3000) / 40)))
    chr_s = min(100, int(float(charger) * 1.1))
    return int(mah_s * 0.55 + chr_s * 0.45)

def _skor_kamera(kamera_str):
    return min(100, int(_ekstrak_mp(kamera_str) * 1.4))

def _skor_total(p, l, b, k):
    return int(p * 0.30 + l * 0.28 + b * 0.22 + k * 0.20)

def hitung_skor_detail(row, req_harga, req_ram, req_memori, req_baterai):
    s_h = 100 if row['Harga_Num'] <= req_harga else max(0, 100 - (row['Harga_Num'] - req_harga) / req_harga * 100)
    s_r = 100 if row['RAM'] >= req_ram          else (row['RAM'] / req_ram) * 100
    s_m = 100 if row['Memori_Internal'] >= req_memori else (row['Memori_Internal'] / req_memori) * 100
    if req_baterai and req_baterai > 0:
        s_b = 100 if row['Baterai'] >= req_baterai else (row['Baterai'] / req_baterai) * 100
    else:
        s_b = 100
    return {"harga": round(s_h, 1), "ram": round(s_r, 1), "memori": round(s_m, 1), "baterai": round(s_b, 1)}

# ==========================================
# MODELS
# ==========================================
class SpesifikasiUser(BaseModel):
    budget:       int
    ram:          int
    memori:       int
    baterai:      Optional[int] = 0
    brand:        Optional[str] = ""
    chipset_skor: Optional[int] = 2
    umur_hp:      Optional[int] = 0

class BandingkanRequest(BaseModel):
    nama_list: List[str]

# ==========================================
# ENDPOINT: HEALTH CHECK
# ==========================================
@app.get("/api/ping")
def ping():
    return {"status": "ok", "total_hp": len(df_katalog)}

# ==========================================
# ENDPOINT: REKOMENDASI AI (Top 3)
# ==========================================
@app.post("/api/rekomendasi")
def dapatkan_rekomendasi(data: SpesifikasiUser):
    # FIX #4 — Toleransi +200.000 disamakan dengan app_offline.py
    mask  = df_katalog['Harga_Num'] <= (data.budget + 200000)
    df_f  = df_katalog[mask].copy()
    total = len(df_f)

    if df_f.empty:
        return {"status": "gagal", "pesan": "Budget tidak mencukupi untuk HP di database."}

    vek_cols = [c for c in df_vektor.columns if not c.startswith('_')]
    vek_f    = df_vektor.loc[mask, vek_cols]

    user_vec = pd.DataFrame(0, index=[0], columns=vek_cols)
    if 'Harga_Angka'     in user_vec.columns: user_vec.at[0, 'Harga_Angka']     = data.budget
    if 'RAM'             in user_vec.columns: user_vec.at[0, 'RAM']             = data.ram
    if 'Memori_Internal' in user_vec.columns: user_vec.at[0, 'Memori_Internal'] = data.memori
    if 'Chipset_Skor'    in user_vec.columns: user_vec.at[0, 'Chipset_Skor']    = data.chipset_skor
    if 'Umur_HP'         in user_vec.columns: user_vec.at[0, 'Umur_HP']         = data.umur_hp
    if data.baterai and data.baterai > 0:
        if 'Baterai' in user_vec.columns: user_vec.at[0, 'Baterai'] = data.baterai
    if data.brand:
        col = f'Brand_{data.brand.upper()}'
        if col in user_vec.columns: user_vec.at[0, col] = 1

    user_scaled  = scaler.transform(user_vec)
    skor_sim     = cosine_similarity(user_scaled, vek_f)
    df_f['Skor'] = skor_sim[0]
    top3         = df_f.sort_values('Skor', ascending=False).head(3)

    rekom_list   = []
    radar_series = []
    all_scores   = []

    for rank, (idx, row) in enumerate(top3.iterrows(), start=1):
        skor_aman = max(0.0, min(1.0, float(row['Skor'])))
        persen    = int(skor_aman * 100)
        sd        = hitung_skor_detail(row, data.budget, data.ram, data.memori, data.baterai or 0)

        # FIX #6 — Layar_Jenis dari kolom yang sudah diinfer di startup
        layar_jenis = str(row.get('Layar_Jenis', 'IPS LCD') or 'IPS LCD')
        try:
            refresh = int(df_vektor.at[idx, '_Refresh_orig'])
        except Exception:
            refresh = 60

        # FIX #7 — Umur_HP dari kolom yang dihitung dari Rilis
        umur_hp = int(row.get('Umur_HP', 0) or 0)

        raw = {
            "nama":        row['Nama'],
            "brand":       str(row.get('Brand', '-') or '-'),
            "rilis":       str(int(_parse_tahun(row.get('Rilis', 2025)))),
            "harga_num":   int(row['Harga_Num']),
            "harga_str":   f"Rp {int(row['Harga_Num']):,}".replace(',', '.'),
            "ram":         int(row.get('RAM', 0)),
            "memori":      int(row.get('Memori_Internal', 0)),
            "baterai":     int(row.get('Baterai', 0)),
            "chipset":     str(row.get('Chipset', '-') or '-'),
            "charger":     str(row.get('Charger', '-') or '-'),
            "kamera":      str(row.get('Kamera_Utama', '-') or '-'),
            "layar_jenis": layar_jenis,
            "refresh":     str(refresh),
            "os":          str(row.get('OS', '-') or '-'),
            "umur":        f"{umur_hp} Tahun",
            "url":         str(row.get('URL', '#') or '#'),
            "persentase":  persen,
        }
        rekom_list.append({"rank": rank, "raw": raw, "skor_detail": sd})
        all_scores.append(raw)

    best   = all_scores[0]
    alasan = []
    if best['baterai'] >= max(h['baterai'] for h in all_scores): alasan.append("Kapasitas Baterai Terbesar")
    if best['ram']     >= max(h['ram']     for h in all_scores): alasan.append("RAM Paling Lega")
    if best['harga_num'] <= min(h['harga_num'] for h in all_scores): alasan.append("Harga Paling Terjangkau")
    teks = "Sistem AI merekomendasikan HP ini karena memiliki keseimbangan fitur terbaik sesuai profil Anda."
    if alasan: teks += f" Keunggulan utamanya: **{', '.join(alasan)}**."

    df_r  = pd.DataFrame({
        'Harga':   [h['harga_num']  for h in all_scores],
        'RAM':     [h['ram']        for h in all_scores],
        'Memori':  [h['memori']     for h in all_scores],
        'Baterai': [h['baterai']    for h in all_scores],
        'Skor AI': [h['persentase'] for h in all_scores],
    })
    sc2   = MinMaxScaler()
    ncols = ['Harga', 'RAM', 'Memori', 'Baterai', 'Skor AI']
    df_rs = pd.DataFrame(sc2.fit_transform(df_r[ncols]), columns=ncols)
    df_rs['Harga'] = 1 - df_rs['Harga']
    for i, row in df_rs.iterrows():
        radar_series.append({
            "name":  all_scores[i]['nama'],
            "r":     [float(row[c]) for c in ncols],
            "theta": ['Keterjangkauan Harga', 'Kapasitas RAM', 'Memori Internal', 'Kapasitas Baterai', 'Total Skor AI'],
        })

    return {
        "status":     "sukses",
        "total":      total,
        "data":       rekom_list,
        "radar":      radar_series,
        "kesimpulan": {"nama": best['nama'], "harga": best['harga_str'], "teks": teks},
    }

# ==========================================
# ENDPOINT: KATALOG
# ==========================================
@app.get("/api/katalog")
def get_katalog():
    nama_list = sorted(df_katalog['Nama'].tolist())
    detail = {
        row['Nama']: {
            "brand":  str(row.get('Brand', '-') or '-'),
            "rilis":  int(_parse_tahun(row.get('Rilis', 2025))),
            "harga":  int(row['Harga_Num']),
            "ram":    int(row.get('RAM', 0)    or 0),
            "memori": int(row.get('Memori_Internal', 0) or 0),
        }
        for _, row in df_katalog.iterrows()
    }
    return {"nama_list": nama_list, "detail": detail}

# ==========================================
# ENDPOINT: PERBANDINGAN HP
# ==========================================
@app.post("/api/bandingkan")
def bandingkan_hp(data: BandingkanRequest):
    if len(data.nama_list) < 2:
        return {"status": "gagal", "pesan": "Pilih minimal 2 HP."}

    result = []
    for nama in data.nama_list:
        matches = df_katalog[df_katalog['Nama'] == nama]
        if matches.empty:
            continue
        idx = matches.index[0]
        row = matches.iloc[0]

        # FIX #8 — Chip_orig dan Refresh dari precomputed df_vektor columns
        chip_orig = 2.0
        refresh   = 60
        try:
            chip_orig = float(df_vektor.at[idx, '_Chip_orig'])
            refresh   = int(df_vektor.at[idx, '_Refresh_orig'])
        except Exception:
            pass

        baterai = int(row.get('Baterai', 4000)       or 4000)
        charger = float(row.get('Charger', 18)        or 18)
        ram     = int(row.get('RAM', 8)               or 8)
        memori  = int(row.get('Memori_Internal', 128) or 128)

        # FIX #9 — Gunakan Layar_Jenis dari kolom yang sudah diinfer di startup
        layar_jenis = str(row.get('Layar_Jenis', 'IPS LCD') or 'IPS LCD')

        # FIX #10 — Umur_HP dari kolom yang dihitung dari Rilis
        umur_hp = int(row.get('Umur_HP', 0) or 0)

        p = _skor_performa(ram, chip_orig)
        l = _skor_layar(layar_jenis, refresh)
        b = _skor_baterai(baterai, charger)
        k = _skor_kamera(str(row.get('Kamera_Utama', '12 MP') or '12 MP'))
        t = _skor_total(p, l, b, k)

        result.append({
            "nama":          row['Nama'],
            "brand":         str(row.get('Brand', '') or ''),
            "rilis":         str(int(_parse_tahun(row.get('Rilis', 2025)))),
            "harga_num":     int(row['Harga_Num']),
            "harga_str":     f"Rp {int(row['Harga_Num']):,}".replace(',', '.'),
            "ram":           ram,
            "memori":        memori,
            "baterai":       baterai,
            "charger":       int(charger),
            "chipset":       str(row.get('Chipset', '-') or '-'),
            "kamera":        str(row.get('Kamera_Utama', '-') or '-'),
            "layar_jenis":   layar_jenis,
            "refresh":       refresh,
            "os":            str(row.get('OS', '-') or '-')[:40],
            "umur_hp":       umur_hp,
            "skor_performa": p,
            "skor_layar":    l,
            "skor_baterai":  b,
            "skor_kamera":   k,
            "skor_total":    t,
        })

    if len(result) < 2:
        return {"status": "gagal", "pesan": "Data HP tidak ditemukan di katalog."}

    best_idx = max(range(len(result)), key=lambda i: result[i]['skor_total'])
    for i, r in enumerate(result):
        r['is_best'] = (i == best_idx)

    best = result[best_idx]
    cat_scores = {
        'Performa':           best['skor_performa'],
        'Layar':              best['skor_layar'],
        'Baterai & Charging': best['skor_baterai'],
        'Kamera':             best['skor_kamera'],
    }
    top2   = sorted(cat_scores.items(), key=lambda x: -x[1])[:2]
    alasan = []
    for cat, _ in top2:
        if cat == 'Performa':
            cw = best['chipset'].split()[0] if best['chipset'] not in ('-', '') else ''
            alasan.append(f"Chipset {cw} dengan RAM {best['ram']}GB sangat responsif")
        elif cat == 'Layar':
            alasan.append(f"Layar {best['layar_jenis']} {best['refresh']}Hz memberikan visual premium")
        elif cat == 'Baterai & Charging':
            alasan.append(f"Baterai {best['baterai']} mAh + pengisian {best['charger']}W sangat efisien")
        elif cat == 'Kamera':
            alasan.append(f"Kamera {_ekstrak_mp(best['kamera'])} MP menghasilkan foto berkualitas tinggi")

    nama_s = best['nama'].split('(')[0].strip()
    teks   = f"{' dan '.join(alasan)}, menjadikan {nama_s} pilihan terbaik di rentang harga {best['harga_str']}."

    return {
        "status":     "sukses",
        "data":       result,
        "best_idx":   best_idx,
        "kesimpulan": {"nama": nama_s, "harga": best['harga_str'], "teks": teks},
    }