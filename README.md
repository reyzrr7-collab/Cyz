# ⚽ Football Analysis Agent — Railway Deployment Guide

Bot Telegram analisis sepak bola berbasis ReAct + NVIDIA Nemotron 120B.

---

## 🚀 Deploy ke Railway

### 1. Push ke GitHub
```bash
git init
git add .
git commit -m "football agent — railway ready"
git remote add origin https://github.com/username/football-agent.git
git push -u origin main
```

### 2. Buat project Railway
1. Buka [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Pilih repository Anda

### 3. Set Environment Variables
Di Railway dashboard → **Settings** → **Variables**, tambahkan:

| Variable | Keterangan | Wajib |
|---|---|---|
| `NVIDIA_API_KEY` | API key NVIDIA NIM | ✅ |
| `TAVILY_API_KEY` | API key Tavily Search | ✅ |
| `TELEGRAM_TOKEN` | Token bot Telegram (dari @BotFather) | ✅ |
| `TELEGRAM_CHAT_ID` | Chat ID Telegram Anda | ✅ |
| `FOOTBALL_API_KEY` | API Football (RapidAPI) | Opsional |
| `ODDS_API_KEY` | The Odds API | Opsional |

### 4. Set Service Type
Di Railway → **Settings** → pastikan `Procfile` terdeteksi:
```
worker: python main.py
```

### 5. Deploy
Railway akan otomatis build dan menjalankan bot.

---

## 💬 Cara Pakai Bot Telegram

| Command | Fungsi |
|---|---|
| `/start` | Sapa bot & lihat instruksi |
| `/fakta` | Lihat profil & fakta tersimpan |
| _(teks bebas)_ | Kirim pertanyaan analisis sepak bola |

Contoh pertanyaan:
- _"Analisis Liverpool vs Arsenal besok"_
- _"Prediksi Real Madrid vs Barcelona akhir pekan ini"_
- _"Berapa xG rata-rata Bayern Munich musim ini?"_

---

## ⚠️ Catatan Penting

### SQLite di Railway (Ephemeral Storage)
Data percakapan (`/tmp/agent_memory.db`) **akan terhapus** setiap kali Railway me-restart atau re-deploy service Anda.

**Solusi permanen:** tambahkan Railway PostgreSQL dan ganti fungsi `init_db()`, `save_to_memory()`, dll. menggunakan `psycopg2` dengan `DATABASE_URL` dari Railway.

### Session ID
Di versi bot Telegram ini, `session_id` = Telegram User ID Anda secara otomatis. Riwayat percakapan tersimpan per user.

---

## 📦 Dependencies
```
python-telegram-bot>=20.7
requests>=2.31.0
aiohttp>=3.9.0
understat>=1.0.1
tavily-python>=0.3.0
```
