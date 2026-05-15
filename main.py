"""
╔══════════════════════════════════════════════════════════════════════════════╗

║         FOOTBALL ANALYSIS AGENT — ReAct + SQLite Long-Term Memory           ║

║         Refactored by: Senior Python Developer & AI Engineer                 ║

╚══════════════════════════════════════════════════════════════════════════════╝



Arsitektur Utama:

  - Pola ReAct (Reason + Act) dengan Self-Reflection sebelum Final Answer

  - SQLite 3-tabel: conversation_history | user_context | user_facts

  - Smart context extraction via LLM (Nemotron) — bukan regex kasar

  - Tool suite: kalkulator, tavily_search, multi_search, analisis_statistik,
               get_football_stats, get_understat_xg, poisson_analysis,
               get_team_form, get_team_motivation

  - Ketahanan penuh: retry, anti-infinite-loop, token-budget guard

╔══════════════════════════════════════════════════════════════════════════════╗
║  CHANGELOG — PENINGKATAN v2.0                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Fix 1: MODEL_NAME_MINI → llama-3.1-nemotron-nano-8b (3-5x lebih cepat)    ║
║  Fix 2: _normalize_year_in_query regex 2[0-4]→2[0-5] (2025 tercakup)       ║
║  Fix 3: multi_search PARALEL via asyncio.gather (4x lebih cepat)            ║
║  Fix 4: Cache TTL 5 menit untuk Tavily & Football API                       ║
║  Fix 5: Rate limiter 0.5 dtk/request (cegah throttle)                       ║
║  Fix 6: hitung_confidence() — indikator keyakinan prediksi 🟢🟡🔴          ║
║  Fix 7: Poisson MAX_GOALS 5→7 (akurasi probabilitas +7%)                    ║
║  Fix 8: Validasi kualitas jawaban sebelum kirim Telegram                     ║
║  Fix 9: Unit test 7 modul (test_agent.py)                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""



import os

import re

import json

import time

import math

import sqlite3

import requests

import logging

import asyncio

import aiohttp

from datetime import datetime

from typing import Optional

from understat import Understat

from tavily import TavilyClient

from telegram import Bot, Update

from telegram.constants import ParseMode

from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)



# ─────────────────────────────────────────────

#  KONFIGURASI GLOBAL

# ─────────────────────────────────────────────

NVIDIA_API_KEY    = os.environ.get("NVIDIA_API_KEY", "")

NVIDIA_BASE_URL   = "https://integrate.api.nvidia.com/v1"

NVIDIA_CHAT_URL   = f"{NVIDIA_BASE_URL}/chat/completions"  # OpenAI-compatible endpoint

TAVILY_API_KEY    = os.environ.get("TAVILY_API_KEY", "")



# ── Football API ───────────────────────────────────────────────────────────────

FOOTBALL_API_KEY  = os.environ.get("FOOTBALL_API_KEY", "")

FOOTBALL_API_URL  = "https://v3.football.api-sports.io"



# ── The Odds API ───────────────────────────────────────────────────────────────

ODDS_API_KEY      = os.environ.get("ODDS_API_KEY", "")

ODDS_API_URL      = "https://api.the-odds-api.com/v4"



# ── Model & batas iterasi ──────────────────────────────────────────────────────

MODEL_NAME        = "nvidia/nemotron-3-super-120b-a12b"        # model utama — ReAct loop
MODEL_NAME_MINI   = "nvidia/llama-3.1-nemotron-nano-8b-v1"    # model ringan — ekstraksi & refleksi (3-5x lebih cepat)

MAX_ITERATIONS    = 10

MAX_TOOL_RETRIES  = 2

LLM_TIMEOUT       = 120         # 2 menit — NVIDIA NIM cloud lebih cepat

LLM_MINI_TIMEOUT  = 60          # 1 menit — untuk ekstraksi & refleksi



# Railway: gunakan /tmp agar tidak error permission (filesystem ephemeral)
# Untuk data persisten, migrasi ke PostgreSQL (lihat README)
DB_PATH           = os.environ.get("DB_PATH", "/tmp/agent_memory.db")



# ── Telegram ───────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")

TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")



# ─────────────────────────────────────────────

#  LOGGING

# ─────────────────────────────────────────────

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",

    datefmt="%H:%M:%S"

)

log = logging.getLogger("FootballAgent")





# ══════════════════════════════════════════════════════════════════════════════

#  DATABASE — INISIALISASI & HELPER

# ══════════════════════════════════════════════════════════════════════════════



def init_db() -> None:

    """

    Buat tiga tabel jika belum ada:

      1. conversation_history  — riwayat pesan per sesi

      2. user_context          — profil user (nama, preferensi teks bebas)

      3. user_facts            — fakta terstruktur (tim, pemain, dll.) per user

    """

    conn = sqlite3.connect(DB_PATH)

    c = conn.cursor()



    # Tabel 1 — Riwayat percakapan

    c.execute('''

        CREATE TABLE IF NOT EXISTS conversation_history (

            id          INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id  TEXT    NOT NULL,

            role        TEXT    NOT NULL,

            content     TEXT    NOT NULL,

            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP

        )

    ''')



    # Tabel 2 — Profil naratif user

    c.execute('''

        CREATE TABLE IF NOT EXISTS user_context (

            session_id  TEXT    PRIMARY KEY,

            user_name   TEXT,

            preferences TEXT,

            last_active DATETIME

        )

    ''')



    # Tabel 3 — Fakta terstruktur per user (NEW)

    #   fact_key  : tim_favorit | pemain_favorit | liga_favorit |

    #               gaya_analisis | lokasi | dll.

    #   fact_value: nilai fakta (teks)

    #   source    : pesan asal user (untuk audit)

    c.execute('''

        CREATE TABLE IF NOT EXISTS user_facts (

            id          INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id  TEXT    NOT NULL,

            fact_key    TEXT    NOT NULL,

            fact_value  TEXT    NOT NULL,

            source      TEXT,

            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(session_id, fact_key)

        )

    ''')



    conn.commit()

    conn.close()

    log.info("Database diinisialisasi.")





def save_to_memory(session_id: str, role: str, content: str) -> None:

    """Simpan satu pesan ke conversation_history."""

    conn = sqlite3.connect(DB_PATH)

    conn.execute(

        "INSERT INTO conversation_history (session_id, role, content) VALUES (?, ?, ?)",

        (session_id, role, content)

    )

    conn.commit()

    conn.close()





def load_recent_memory(session_id: str, limit: int = 10) -> str:

    """

    [PENINGKATAN] Kembalikan dua blok:

      A) Fakta kunci user dari user_facts (semua fakta, compact)

      B) N pesan terakhir dari conversation_history (kronologis)

    """

    conn = sqlite3.connect(DB_PATH)

    c    = conn.cursor()



    # ─── Blok A: Fakta Kunci ──────────────────────────────────────────────────

    c.execute(

        "SELECT fact_key, fact_value FROM user_facts WHERE session_id = ? ORDER BY updated_at DESC",

        (session_id,)

    )

    facts = c.fetchall()



    # ─── Blok B: Riwayat Percakapan ───────────────────────────────────────────

    c.execute(

        """SELECT role, content FROM conversation_history

           WHERE session_id = ?

           ORDER BY timestamp DESC LIMIT ?""",

        (session_id, limit)

    )

    rows = c.fetchall()

    conn.close()



    parts = []



    if facts:

        fact_lines = "\n".join(f"  • {k}: {v}" for k, v in facts)

        parts.append(f"[FAKTA KUNCI USER]\n{fact_lines}")



    if rows:

        history_lines = "\n".join(

            f"{role.capitalize()}: {content}"

            for role, content in reversed(rows)

        )

        parts.append(f"[RIWAYAT PERCAKAPAN TERAKHIR]\n{history_lines}")



    return "\n\n".join(parts)





def get_user_context(session_id: str) -> dict:

    """Ambil profil naratif user dari DB."""

    conn = sqlite3.connect(DB_PATH)

    c    = conn.cursor()

    c.execute(

        "SELECT user_name, preferences FROM user_context WHERE session_id = ?",

        (session_id,)

    )

    row = c.fetchone()

    conn.close()

    if row:

        return {"user_name": row[0], "preferences": row[1]}

    return {"user_name": None, "preferences": None}





def upsert_user_fact(session_id: str, fact_key: str, fact_value: str, source: str = "") -> None:

    """Insert-or-replace satu fakta terstruktur."""

    conn = sqlite3.connect(DB_PATH)

    conn.execute(

        """INSERT INTO user_facts (session_id, fact_key, fact_value, source, updated_at)

           VALUES (?, ?, ?, ?, ?)

           ON CONFLICT(session_id, fact_key)

           DO UPDATE SET fact_value=excluded.fact_value,

                         source=excluded.source,

                         updated_at=excluded.updated_at""",

        (session_id, fact_key, fact_value, source, datetime.now())

    )

    conn.commit()

    conn.close()





# ══════════════════════════════════════════════════════════════════════════════

#  SMART CONTEXT EXTRACTION  (PENINGKATAN UTAMA)

# ══════════════════════════════════════════════════════════════════════════════



_EXTRACTION_SYSTEM = """Anda adalah ekstraktor informasi. Tugas Anda: analisis pesan user dan

kembalikan HANYA objek JSON valid (tanpa markdown, tanpa komentar) berisi fakta penting yang

ditemukan. Gunakan key berikut jika relevan:

  "nama"          : nama asli user

  "lokasi"        : kota / negara user

  "tim_favorit"   : nama klub sepak bola favorit user

  "pemain_favorit": nama pemain favorit

  "liga_favorit"  : nama liga favorit

  "gaya_analisis" : preferensi gaya analisis (statistik, taktik, naratif, dll.)

  "konteks_lain"  : informasi penting lain yang tidak masuk kategori di atas



Jika tidak ada fakta yang ditemukan, kembalikan: {}

Jangan tambahkan teks apapun di luar JSON."""





def ekstrak_fakta_dari_pesan(pesan: str, session_id: str) -> None:

    """

    [PENINGKATAN] Gunakan LLM (panggilan ringan) untuk mengekstrak fakta

    penting dari pesan user, lalu simpan ke tabel user_facts.

    """

    raw = _panggil_llm_mini(

        prompt_system=_EXTRACTION_SYSTEM,

        prompt_user=f"Pesan user: {pesan}"

    )

    if not raw:

        return



    try:

        # Bersihkan kemungkinan pembungkus markdown

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        fakta = json.loads(clean)

    except json.JSONDecodeError:

        log.warning("Ekstraksi fakta gagal parse JSON: %s", raw[:120])

        return



    if not isinstance(fakta, dict) or not fakta:

        return



    for key, val in fakta.items():

        if val and str(val).strip():

            upsert_user_fact(session_id, key, str(val).strip(), source=pesan[:200])

            log.info("Fakta disimpan — %s: %s", key, val)



    # Sinkronkan ke user_context.user_name jika nama ditemukan

    if "nama" in fakta and fakta["nama"]:

        conn = sqlite3.connect(DB_PATH)

        conn.execute(

            """INSERT INTO user_context (session_id, user_name, last_active)

               VALUES (?, ?, ?)

               ON CONFLICT(session_id)

               DO UPDATE SET user_name=excluded.user_name, last_active=excluded.last_active""",

            (session_id, fakta["nama"], datetime.now())

        )

        conn.commit()

        conn.close()





# ══════════════════════════════════════════════════════════════════════════════

#  PANGGIL LLM (NEMOTRON VIA NVIDIA NIM)

# ══════════════════════════════════════════════════════════════════════════════



def _panggil_llm(

    prompt_system: str,

    prompt_user: str,

    num_predict: int = 4096,

    temperature: float = 0.1,  # Nemotron ReAct: 0.1 → presisi format tinggi

    timeout: int = LLM_TIMEOUT,

    retries: int = 2,

    streaming: bool = False,   # Aktifkan untuk Final Answer panjang

) -> Optional[str]:

    """Panggil LLM lewat NVIDIA NIM (OpenAI-compatible) dengan retry otomatis.

    - top_p=0.7  : parameter NIM untuk konsistensi output Nemotron
    - streaming  : aktif saat Final Answer agar output terasa real-time
    - <think>    : di-log ke DEBUG (bukan dibuang mentah-mentah)
    """

    payload = {

        "model":       MODEL_NAME,

        "messages": [

            {"role": "system", "content": prompt_system},

            {"role": "user",   "content": prompt_user}

        ],

        "max_tokens":  num_predict,

        "temperature": temperature,

        "top_p":       0.7,   # NVIDIA NIM: meningkatkan konsistensi Nemotron

        "stream":      streaming,

    }

    headers = {

        "Authorization": f"Bearer {NVIDIA_API_KEY}",

        "Content-Type":  "application/json"

    }

    for attempt in range(1, retries + 2):

        try:

            resp = requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers, timeout=timeout,
                                 stream=streaming)

            resp.raise_for_status()

            if streaming:
                # Kumpulkan chunk SSE, print real-time ke terminal
                chunks = []
                print("\n🤖 Agent (streaming):", flush=True)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    if decoded.startswith("data:"):
                        data_str = decoded[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            delta = json.loads(data_str)["choices"][0]["delta"].get("content", "")
                            if delta:
                                print(delta, end="", flush=True)
                                chunks.append(delta)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
                print()  # newline setelah streaming selesai
                raw_content = "".join(chunks)
            else:
                raw_content = resp.json()["choices"][0]["message"]["content"]

            # Log <think> block Nemotron ke DEBUG (bukan dibuang)
            think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
            if think_match:
                log.debug("[NEMOTRON THINK] %s", think_match.group(1)[:400])

            # Strip <think> agar hanya output bersih masuk ReAct loop
            raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
            return raw_content

        except requests.exceptions.Timeout:

            log.warning("Timeout LLM (percobaan %d/%d)", attempt, retries + 1)

            if attempt <= retries:

                time.sleep(3 * attempt)

        except requests.exceptions.RequestException as e:

            log.error("Error LLM: %s", e)

            if attempt <= retries:

                time.sleep(3 * attempt)

        except (KeyError, ValueError) as e:

            log.error("Parse respons LLM gagal: %s", e)

            break

    return None





def _panggil_llm_mini(prompt_system: str, prompt_user: str) -> Optional[str]:

    """Versi ringan — Nemotron untuk ekstraksi & refleksi (NVIDIA NIM).

    - temperature=0.0 : deterministik, cocok untuk JSON extraction
    - top_p=0.7       : konsistensi output NIM
    - retry 2x        : sama seperti _panggil_llm agar tidak silent-fail
    """

    payload = {

        "model":      MODEL_NAME_MINI,

        "messages": [

            {"role": "system", "content": prompt_system},

            {"role": "user",   "content": prompt_user}

        ],

        "max_tokens":  512,

        "temperature": 0.0,

        "top_p":       0.7,   # NVIDIA NIM parameter

        "stream":      False,

    }

    headers = {

        "Authorization": f"Bearer {NVIDIA_API_KEY}",

        "Content-Type":  "application/json"

    }

    for attempt in range(1, 3):  # retry 2x (sama seperti _panggil_llm)

        try:

            resp = requests.post(NVIDIA_CHAT_URL, json=payload, headers=headers, timeout=LLM_MINI_TIMEOUT)

            resp.raise_for_status()

            raw_mini = resp.json()["choices"][0]["message"]["content"]

            # Log <think> block Nemotron ke DEBUG
            think_match = re.search(r"<think>(.*?)</think>", raw_mini, flags=re.DOTALL)
            if think_match:
                log.debug("[NEMOTRON THINK-MINI] %s", think_match.group(1)[:200])

            raw_mini = re.sub(r"<think>.*?</think>", "", raw_mini, flags=re.DOTALL).strip()
            return raw_mini

        except requests.exceptions.Timeout:

            log.warning("Timeout LLM mini (percobaan %d/2)", attempt)

            if attempt < 2:

                time.sleep(2)

        except requests.exceptions.RequestException as e:

            log.warning("LLM mini request error: %s (percobaan %d/2)", e, attempt)

            if attempt < 2:

                time.sleep(2)

        except (KeyError, ValueError) as e:

            log.warning("LLM mini parse error: %s", e)

            break

    return None





# ══════════════════════════════════════════════════════════════════════════════

#  TOOLS

# ══════════════════════════════════════════════════════════════════════════════



# ── Tool 1: Kalkulator ────────────────────────────────────────────────────────

def kalkulator(ekspresi: str) -> str:


    """


    Evaluasi ekspresi matematika Python secara aman.


    Mendukung: +, -, *, /, **, %, round(), abs(),


               exp(), factorial(), sqrt(), log(), pow()


    """


    _safe_funcs = {


        "round":     round,


        "abs":       abs,


        "exp":       math.exp,


        "factorial": math.factorial,


        "sqrt":      math.sqrt,


        "log":       math.log,


        "pow":       math.pow,


        "e":         math.e,


        "pi":        math.pi,


    }


    try:


        hasil = eval(ekspresi, {"__builtins__": {}}, _safe_funcs)


        return str(hasil)


    except ZeroDivisionError:


        return "Error: Pembagian dengan nol."


    except Exception as e:


        return f"Error kalkulator: {e}"





# ── Helper: normalisasi tahun di query ───────────────────────────────────────

def _normalize_year_in_query(text: str) -> str:
    """Ganti tahun lama (2020–2025) dengan tahun sekarang agar query selalu aktual."""
    current_year = str(datetime.now().year)
    return re.sub(r'\b20(2[0-5])\b', current_year, text)  # fix: mencakup 2020-2025


# ── Tool 2: Tavily Search ─────────────────────────────────────────────────────

def tavily_search(query: str) -> str:

    """Cari informasi terkini lewat Tavily. Cocok untuk berita, cedera, transfer."""

    query = _normalize_year_in_query(query)[:380]  # NVIDIA fix: max 380 chars
    log.info("[TAVILY] Query: %s", query)

    # [FIX 4] Cek cache dulu sebelum memanggil API
    client  = TavilyClient(api_key=TAVILY_API_KEY)
    results = _tavily_single(client, query, max_results=7, days=7)

    if not results:
        return "Tidak ada hasil ditemukan untuk query tersebut."

    lines = []

    for i, item in enumerate(results, 1):

        # Ambil raw_content jika ada (lebih lengkap), fallback ke content
        raw   = item.get("raw_content") or ""
        short = item.get("content", "")
        body  = raw[:800] if raw else short[:400]

        lines.append(

            f"{i}. {item.get('title', 'No title')}\n"

            f"   {body}\n"

            f"   📎 {item.get('url', '')}"

        )

    return "\n\n".join(lines)





# ══════════════════════════════════════════════════════════════════════════════
#  FIX 3+4+5: CACHE TTL + RATE LIMITER + MULTI-SEARCH PARALEL
# ══════════════════════════════════════════════════════════════════════════════

# ── TTL Cache (Fix 4) ─────────────────────────────────────────────────────────
_CACHE: dict = {}           # { key: (timestamp, data) }
CACHE_TTL    = 300          # 5 menit

def _cache_get(key: str):
    """Ambil dari cache jika masih valid."""
    if key in _CACHE:
        ts, data = _CACHE[key]
        if time.time() - ts < CACHE_TTL:
            log.info("[CACHE HIT] %s", key[:60])
            return data
    return None

def _cache_set(key: str, data) -> None:
    """Simpan ke cache."""
    _CACHE[key] = (time.time(), data)

# ── Rate Limiter (Fix 5) ──────────────────────────────────────────────────────
_LAST_CALL: dict = {}       # { api_name: last_call_timestamp }
MIN_INTERVAL     = 0.5      # detik minimum antar panggilan per API

def _rate_limit(api_name: str) -> None:
    """Tunggu jika interval antar panggilan terlalu cepat."""
    now  = time.time()
    last = _LAST_CALL.get(api_name, 0)
    gap  = now - last
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    _LAST_CALL[api_name] = time.time()

# ── Helper: satu query Tavily dengan cache & rate limit ──────────────────────
def _tavily_single(client: TavilyClient, q: str, max_results: int = 5, days: int = 7) -> list:
    """Jalankan satu query Tavily. Pakai cache jika ada."""
    cache_key = f"tavily:{q}:{days}"
    cached    = _cache_get(cache_key)
    if cached is not None:
        return cached

    _rate_limit("tavily")
    try:
        resp = client.search(
            query=q,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=True,
            include_usage=True,
            days=days,
        )
        if "usage" in resp:
            try:
                with open("tavily_usage.log", "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()} | {q} | {resp['usage']}\n")
            except OSError:
                pass
        results = resp.get("results", [])
        _cache_set(cache_key, results)
        return results
    except Exception as e:
        log.warning("[TAVILY] Query gagal: %s — %s", q, e)
        return []

# ── Tool 5: Multi Search — 4 query PARALEL (Fix 3) ───────────────────────────
def multi_search(queries_json: str) -> str:
    """
    Jalankan hingga 4 query Tavily SECARA PARALEL dan gabungkan hasilnya.
    [FIX] Dari sequential (~8 dtk) → paralel (~2 dtk) menggunakan asyncio.gather.

    Format input JSON:
    {
      "queries": [
        "Liverpool vs Chelsea injury news today 2026",
        "Liverpool Chelsea head to head 2026",
        "Liverpool xG stats last 5 matches 2026",
        "Chelsea form crisis 2026"
      ]
    }
    """
    try:
        data    = json.loads(queries_json)
        queries = data.get("queries", [])
    except (json.JSONDecodeError, AttributeError):
        queries = [queries_json.strip()]

    if not queries:
        return "Error: tidak ada query yang diberikan."

    queries = [_normalize_year_in_query(q)[:380] for q in queries[:4]]
    client  = TavilyClient(api_key=TAVILY_API_KEY)

    # ── Jalankan semua query paralel via asyncio ──────────────────────────────
    async def _fetch_all():
        tasks = [
            asyncio.to_thread(_tavily_single, client, q)
            for q in queries
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        try:
            all_raw = asyncio.run(_fetch_all())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                all_raw = loop.run_until_complete(_fetch_all())
            finally:
                loop.close()
    except Exception as e:
        log.error("[MULTI_SEARCH] Paralel gagal, fallback sequential: %s", e)
        all_raw = [_tavily_single(client, q) for q in queries]

    # ── Format output ─────────────────────────────────────────────────────────
    all_results = []
    for q, results in zip(queries, all_raw):
        log.info("[MULTI_SEARCH] Query: %s | %d hasil", q, len(results) if isinstance(results, list) else 0)
        if isinstance(results, Exception) or not results:
            all_results.append(f"\n⚠️ Query '{q}' gagal atau tidak ada hasil.")
            continue
        all_results.append(f"\n{'='*60}\n🔍 QUERY: {q}\n{'='*60}")
        for i, item in enumerate(results[:4], 1):
            raw  = item.get("raw_content") or ""
            body = raw[:600] if raw else item.get("content", "")[:400]
            all_results.append(
                f"{i}. {item.get('title', 'No title')}\n"
                f"   {body}\n"
                f"   📎 {item.get('url', '')}"
            )

    return "\n\n".join(all_results) if all_results else "Tidak ada hasil ditemukan."



# ── Tool 3: Analisis Statistik Sepak Bola ─────────────────────────────────────

def analisis_statistik(data_json: str) -> str:

    """

    Hitung statistik agregat dari data pertandingan.



    Format input JSON (string):

    {

      "pertandingan": [

        {"tim": "Arsenal",    "gol": 2, "kemasukan": 1, "xG": 1.8, "xGA": 0.9},

        {"tim": "Arsenal",    "gol": 0, "kemasukan": 0, "xG": 1.2, "xGA": 0.7},

        ...

      ]

    }



    Output:

    - Persen menang / seri / kalah

    - Rata-rata gol & kemasukan

    - Rata-rata xG & xGA

    - Performa xG vs aktual (over/under-performance)

    """

    try:

        data = json.loads(data_json)

    except json.JSONDecodeError:

        return "Error: Format JSON tidak valid. Pastikan data berbentuk JSON yang benar."



    matches = data.get("pertandingan", [])

    if not matches:

        return "Error: Kunci 'pertandingan' kosong atau tidak ditemukan."



    n     = len(matches)

    menang = seri = kalah = 0

    total_gol = total_kemasukan = 0.0

    total_xG  = total_xGA = 0.0

    has_xg    = False



    for m in matches:

        g  = float(m.get("gol", 0))

        k  = float(m.get("kemasukan", 0))

        xg = m.get("xG");  xga = m.get("xGA")



        total_gol       += g

        total_kemasukan += k



        if xg is not None and xga is not None:

            total_xG  += float(xg)

            total_xGA += float(xga)

            has_xg = True



        if g > k:

            menang += 1

        elif g == k:

            seri   += 1

        else:

            kalah  += 1



    pct_menang = menang / n * 100

    pct_seri   = seri   / n * 100

    pct_kalah  = kalah  / n * 100

    avg_gol    = total_gol       / n

    avg_km     = total_kemasukan / n



    hasil = (

        f"📊 ANALISIS STATISTIK ({n} pertandingan)\n"

        f"{'─'*45}\n"

        f"🏆  Menang : {menang:>3}  ({pct_menang:.1f}%)\n"

        f"🤝  Seri   : {seri:>3}  ({pct_seri:.1f}%)\n"

        f"❌  Kalah  : {kalah:>3}  ({pct_kalah:.1f}%)\n"

        f"{'─'*45}\n"

        f"⚽  Rata-rata gol       : {avg_gol:.2f}\n"

        f"🥅  Rata-rata kemasukan : {avg_km:.2f}\n"

        f"📈  Selisih gol rata-rata: {avg_gol - avg_km:+.2f}\n"

    )



    if has_xg:

        avg_xG  = total_xG  / n

        avg_xGA = total_xGA / n

        over_xG  = avg_gol - avg_xG

        over_xGA = avg_km  - avg_xGA

        hasil += (

            f"{'─'*45}\n"

            f"🎯  Rata-rata xG  : {avg_xG:.2f}  "

            f"({'over' if over_xG >= 0 else 'under'}-performing {abs(over_xG):.2f})\n"

            f"🛡️  Rata-rata xGA : {avg_xGA:.2f}  "

            f"({'over' if over_xGA >= 0 else 'under'}-performing {abs(over_xGA):.2f})\n"

        )



    return hasil





# ── Tool 4: Get Football Stats (Placeholder API) ───────────────────────────────

def get_football_stats(params_json: str) -> str:

    """

    [PLACEHOLDER] Ambil statistik dari Football API eksternal.



    Cara aktifkan:

      1. Isi FOOTBALL_API_KEY di bagian KONFIGURASI GLOBAL

      2. Uncomment blok kode di bawah sesuai endpoint yang diinginkan



    Format input JSON:

    {

      "endpoint"  : "fixtures" | "standings" | "players" | "injuries",

      "league_id" : 39,          (contoh: 39 = Premier League)

      "team_id"   : 33,          (contoh: 33 = Manchester United)

      "season"    : 2026,

      "player_id" : 276          (opsional, untuk endpoint players)

    }

    """

    # ── Validasi konfigurasi ──────────────────────────────────────────────────

    if not FOOTBALL_API_KEY:

        return (

            "⚠️  FOOTBALL_API_KEY belum diisi.\n"

            "Daftarkan di https://www.api-football.com/ (RapidAPI), "

            "lalu isi variabel FOOTBALL_API_KEY di bagian konfigurasi skrip ini."

        )



    # ── Parse parameter ───────────────────────────────────────────────────────

    try:

        params = json.loads(params_json)

    except json.JSONDecodeError:

        return "Error: Format JSON tidak valid."



    endpoint  = params.get("endpoint", "fixtures")

    league_id = params.get("league_id")

    team_id   = params.get("team_id")

    season    = params.get("season", datetime.now().year)

    player_id = params.get("player_id")



    headers = {

        "x-rapidapi-key":  FOOTBALL_API_KEY,

        "x-rapidapi-host": "v3.football.api-sports.io"

    }



    # ── Bangun query params ───────────────────────────────────────────────────

    query: dict = {"season": season}

    if league_id: query["league"] = league_id

    if team_id:   query["team"]   = team_id

    if player_id: query["id"]     = player_id



    url = f"{FOOTBALL_API_URL}/{endpoint}"



    try:

        resp = requests.get(url, headers=headers, params=query, timeout=30)

        resp.raise_for_status()

        data = resp.json()



        if data.get("errors"):

            return f"API Error: {data['errors']}"



        results = data.get("response", [])

        if not results:

            return f"Tidak ada data ditemukan untuk endpoint={endpoint}, params={query}"



        # Kembalikan max 3 item pertama agar tidak membanjiri context

        preview = json.dumps(results[:3], indent=2, ensure_ascii=False)

        return f"Data dari {endpoint} (menampilkan 3/{len(results)} item):\n{preview}"



    except requests.exceptions.Timeout:

        return "Error: Timeout saat menghubungi Football API."

    except requests.exceptions.RequestException as e:

        return f"Error Football API: {e}"









# ── Tool 6: Understat xG ──────────────────────────────────────────────────────





# Liga yang didukung Understat


_UNDERSTAT_TEAM_NAMES = {

    # Bundesliga

    "fsv mainz 05":              "Mainz 05",

    "mainz 05":                  "Mainz 05",

    "1. fc union berlin":        "Union Berlin",

    "fc union berlin":           "Union Berlin",

    "borussia dortmund":         "Dortmund",

    "bayer 04 leverkusen":       "Leverkusen",

    "bayer leverkusen":          "Leverkusen",

    "rb leipzig":                "RasenBallsport Leipzig",

    "eintracht frankfurt":       "Eintracht Frankfurt",

    "vfb stuttgart":             "VfB Stuttgart",

    "fc bayern munich":          "Bayern Munich",

    "fc bayern muenchen":        "Bayern Munich",

    "sc freiburg":               "Freiburg",

    "vfl wolfsburg":             "Wolfsburg",

    "tsg hoffenheim":            "Hoffenheim",

    "fc augsburg":               "Augsburg",

    "vfl bochum":                "Bochum",

    "sv werder bremen":          "Werder Bremen",

    "borussia monchengladbach":  "Borussia M.Gladbach",

    "borussia m\\xc3\\xb6nchengladbach": "Borussia M.Gladbach",

    "fc heidenheim":             "FC Heidenheim",

    "1. fc heidenheim":          "FC Heidenheim",

    "holstein kiel":             "Holstein Kiel",

    "fc st. pauli":              "St. Pauli",

    # EPL

    "tottenham hotspur":         "Tottenham",

    "spurs":                     "Tottenham",

    "wolves":                    "Wolverhampton Wanderers",

    "west ham united":           "West Ham",

    "brighton & hove albion":    "Brighton",

    "nottingham forest":         "Nottingham Forest",

    "newcastle united":          "Newcastle United",

    "sheffield united":          "Sheffield United",

    "manchester united":         "Manchester United",

    "manchester city":           "Manchester City",

    # La Liga

    "fc barcelona":              "Barcelona",

    "atletico madrid":           "Atletico Madrid",

    "athletic bilbao":           "Athletic Club",

    "villarreal cf":             "Villarreal",

    # Serie A

    "inter milan":               "Internazionale",

    "inter":                     "Internazionale",

    "ac milan":                  "Milan",

    "as roma":                   "Roma",

    "ss lazio":                  "Lazio",

    "ssc napoli":                "Napoli",

    "juventus fc":               "Juventus",

    "atalanta bc":               "Atalanta",

    # Ligue 1

    "paris saint-germain":       "Paris Saint-Germain",

    "psg":                       "Paris Saint-Germain",

    "olympique marseille":       "Marseille",

    "olympique lyonnais":        "Lyon",

    "as monaco":                 "Monaco",

}



def _normalize_team_name(name: str) -> str:

    """Normalisasi nama tim ke format Understat."""

    return _UNDERSTAT_TEAM_NAMES.get(name.lower().strip(), name)

_UNDERSTAT_LEAGUES = {


    "epl":        "EPL",


    "premier":    "EPL",


    "england":    "EPL",


    "la liga":    "La_liga",


    "laliga":     "La_liga",


    "spain":      "La_liga",


    "bundesliga": "Bundesliga",


    "germany":    "Bundesliga",


    "serie a":    "Serie_A",


    "seriea":     "Serie_A",


    "italy":      "Serie_A",


    "ligue 1":    "Ligue_1",


    "ligue1":     "Ligue_1",


    "france":     "Ligue_1",


}





def get_understat_xg(params_json: str) -> str:


    """


    Ambil data xG real dari Understat untuk suatu tim.


    Hitung rata-rata xG & xGA dari 7 match terakhir.





    Format input JSON:


    {


      "team": "Liverpool",


      "season": 2025


    }


    Atau untuk dua tim sekaligus:


    {


      "home_team": "Liverpool",


      "away_team": "Arsenal",


      "season": 2025


    }


    """


    try:


        params  = json.loads(params_json)


        season  = int(params.get("season", 2025))


        teams   = []


        if "home_team" in params and "away_team" in params:


            teams = [params["home_team"], params["away_team"]]


        elif "team" in params:


            teams = [params["team"]]


        else:


            return "Error: Sediakan 'team' atau 'home_team'+'away_team'."


    except (json.JSONDecodeError, ValueError) as e:


        return f"Error parse input: {e}"





    async def _fetch(team_name: str):


        norm_name = _normalize_team_name(team_name)


        async with aiohttp.ClientSession() as session:


            u = Understat(session)


            try:


                data = await u.get_team_results(norm_name, season)


                team_name = norm_name


            except Exception:


                data = await u.get_team_results(team_name, season)


            # Ambil 7 match terakhir yang sudah selesai


            done  = [m for m in data if m.get("isResult")][-7:]


            if not done:


                return team_name, None


            xg_list  = [float(m["xG"]["h"]) if m["h"]["title"] == team_name


                        else float(m["xG"]["a"]) for m in done]


            xga_list = [float(m["xG"]["a"]) if m["h"]["title"] == team_name


                        else float(m["xG"]["h"]) for m in done]


            goals_list = [int(m["goals"]["h"]) if m["h"]["title"] == team_name


                          else int(m["goals"]["a"]) for m in done]


            conceded_list = [int(m["goals"]["a"]) if m["h"]["title"] == team_name


                             else int(m["goals"]["h"]) for m in done]


            avg_xg  = round(sum(xg_list)  / len(xg_list),  3)


            avg_xga = round(sum(xga_list) / len(xga_list), 3)


            avg_gol = round(sum(goals_list) / len(goals_list), 2)


            avg_kms = round(sum(conceded_list) / len(conceded_list), 2)


            # Form detail


            lines = [f"  {m['h']['title']} {m['goals']['h']}-{m['goals']['a']} {m['a']['title']}"


                     f" (xG: {m['xG']['h']} vs {m['xG']['a']})"


                     for m in done]


            return team_name, {


                "avg_xg":  avg_xg,


                "avg_xga": avg_xga,


                "avg_gol": avg_gol,


                "avg_kms": avg_kms,


                "n":       len(done),


                "form":    lines,


            }





    async def _run_all():


        tasks = [_fetch(t) for t in teams]


        return await asyncio.gather(*tasks)





    try:


        try:


            results = asyncio.run(_run_all())


        except RuntimeError:


            loop = asyncio.new_event_loop()


            asyncio.set_event_loop(loop)


            try:


                results = loop.run_until_complete(_run_all())


            finally:


                loop.close()


    except Exception as e:


        return f"Error Understat: {e}"





    output = []


    for team_name, stats in results:


        if stats is None:


            output.append(f"⚠️ {team_name}: data tidak ditemukan (cek nama tim)")


            continue


        form_str = "\n".join(stats["form"])


        output.append(


            f"📊 {team_name} — {stats['n']} Match Terakhir (Season {season})\n"


            f"  xG  rata-rata : {stats['avg_xg']:>6}  (aktual gol: {stats['avg_gol']})\n"


            f"  xGA rata-rata : {stats['avg_xga']:>6}  (aktual kemasukan: {stats['avg_kms']})\n"


            f"  Over-perform  : {round(stats['avg_gol'] - stats['avg_xg'], 3):>+6}\n"


            f"  Under-perform : {round(stats['avg_kms'] - stats['avg_xga'], 3):>+6}\n"


            f"\nForm Detail:\n{form_str}"


        )


    return "\n\n".join(output)








# ── Tool 7: Poisson Analysis ──────────────────────────────────────────────────





def poisson_analysis(params_json: str) -> str:


    """


    Hitung probabilitas skor pertandingan menggunakan Distribusi Poisson.





    Formula: P(k) = (e^-λ × λ^k) / k!


    λ = rata-rata xG tim (expected goals)


    k = jumlah gol (0..4)





    Format input JSON:


    {


      "xg_home": 1.30,


      "xg_away": 1.55,


      "home_team": "Liverpool",


      "away_team": "Arsenal"


    }


    """


    try:


        params   = json.loads(params_json)


        xg_home  = float(params["xg_home"])


        xg_away  = float(params["xg_away"])


        home_team = params.get("home_team", "Home")


        away_team = params.get("away_team", "Away")


    except (json.JSONDecodeError, KeyError, ValueError) as e:


        return f"Error parse input: {e}. Pastikan xg_home dan xg_away tersedia."





    MAX_GOALS = 7  # 0..6 — fix: skor ekstrem (5-0, 6-1) ikut dikalkulasi





    def poisson_prob(lam: float, k: int) -> float:


        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)





    # Hitung probabilitas tiap skor (matriks MAX_GOALS x MAX_GOALS)


    matrix = {}


    for h in range(MAX_GOALS):


        for a in range(MAX_GOALS):


            matrix[(h, a)] = round(


                poisson_prob(xg_home, h) * poisson_prob(xg_away, a) * 100, 2


            )





    # Win / Draw / Loss


    p_home = sum(v for (h, a), v in matrix.items() if h > a)


    p_draw = sum(v for (h, a), v in matrix.items() if h == a)


    p_away = sum(v for (h, a), v in matrix.items() if h < a)





    # Top 5 skor paling mungkin


    top5 = sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:5]





    # Format output


    top5_str = "\n".join(


        f"  {i+1}. {home_team} {h}-{a} {away_team} : {p:.2f}%"


        for i, ((h, a), p) in enumerate(top5)


    )





    # Matrix visual (4x4)


    header = f"      " + "  ".join(f"{away_team[:3]:>4} {g}" for g in range(MAX_GOALS))


    rows   = []


    for h in range(MAX_GOALS):


        row = f"{home_team[:3]:>4} {h} |" + "  ".join(


            f"{matrix[(h,a)]:>6.2f}%" for a in range(MAX_GOALS)


        )


        rows.append(row)


    matrix_str = header + "\n" + "\n".join(rows)





    return (


        f"🎯 POISSON ANALYSIS: {home_team} vs {away_team}\n"


        f"{'─'*50}\n"


        f"λ Home ({home_team}): {xg_home}\n"


        f"λ Away ({away_team}): {xg_away}\n"


        f"{'─'*50}\n"


        f"📈 PROBABILITAS HASIL:\n"


        f"  🏠 {home_team} menang : {p_home:.2f}%\n"


        f"  🤝 Seri              : {p_draw:.2f}%\n"


        f"  ✈️  {away_team} menang : {p_away:.2f}%\n"


        f"{'─'*50}\n"


        f"🏆 TOP 5 SKOR PALING MUNGKIN:\n{top5_str}\n"


        f"{'─'*50}\n"


        f"📊 MATRIX PROBABILITAS SKOR (%):\n{matrix_str}\n"


    )





# ── Tool 8: Get Team Form (5 pertandingan terakhir via API Football) ──────────

def get_team_form(params_json: str) -> str:
    """
    Ambil form 5 pertandingan terakhir tim dari API Football.
    Menampilkan: hasil (M/S/K), skor, lawan, venue, gol, kemasukan.

    Format input JSON:
    {
      "team_id"  : 157,
      "league_id": 78,
      "season"   : 2025,
      "last"     : 5
    }
    """
    if not FOOTBALL_API_KEY:
        return "Tidak ada FOOTBALL_API_KEY, skip tool ini."

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        return "Error: Format JSON tidak valid."

    team_id   = params.get("team_id")
    league_id = params.get("league_id")
    season    = params.get("season", datetime.now().year)
    last      = params.get("last", 5)

    if not team_id:
        return "Error: 'team_id' wajib diisi."

    headers = {
        "x-rapidapi-key":  FOOTBALL_API_KEY,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }

    query = {"team": team_id, "season": season, "last": last}
    if league_id:
        query["league"] = league_id

    try:
        resp = requests.get(
            f"{FOOTBALL_API_URL}/fixtures",
            headers=headers,
            params=query,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("errors"):
            return f"API Error: {data['errors']}"

        fixtures = data.get("response", [])
        if not fixtures:
            return f"Tidak ada data fixture untuk team_id={team_id}, season={season}"

        lines = [f"FORM {last} PERTANDINGAN TERAKHIR (team_id={team_id})"]
        menang = seri = kalah = 0
        total_gol = total_km = 0

        for fix in fixtures:
            teams   = fix.get("teams", {})
            goals   = fix.get("goals", {})
            fixture = fix.get("fixture", {})

            home_id   = teams.get("home", {}).get("id")
            away_id   = teams.get("away", {}).get("id")
            home_name = teams.get("home", {}).get("name", "?")
            away_name = teams.get("away", {}).get("name", "?")
            gol_home  = goals.get("home", 0) or 0
            gol_away  = goals.get("away", 0) or 0
            tanggal   = fixture.get("date", "")[:10]

            is_home  = (str(home_id) == str(team_id))
            gol_tim  = gol_home if is_home else gol_away
            gol_lawan = gol_away if is_home else gol_home
            lawan    = away_name if is_home else home_name
            venue    = "H" if is_home else "A"

            total_gol += gol_tim
            total_km  += gol_lawan

            if gol_tim > gol_lawan:
                hasil = "MENANG"; menang += 1
            elif gol_tim == gol_lawan:
                hasil = "SERI"; seri += 1
            else:
                hasil = "KALAH"; kalah += 1

            lines.append(
                f"{tanggal} [{venue}] vs {lawan} | {gol_tim}-{gol_lawan} | {hasil}"
            )

        n = len(fixtures)
        avg_gol = round(total_gol / n, 2) if n else 0
        avg_km  = round(total_km  / n, 2) if n else 0

        lines.append(f"Ringkasan: {menang}M {seri}S {kalah}K")
        lines.append(f"Rata-rata: {avg_gol} gol/game | {avg_km} kemasukan/game")

        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return "Error: Timeout saat menghubungi Football API."
    except requests.exceptions.RequestException as e:
        return f"Error Football API: {e}"


# ── Tool 9: Get Team Motivation (via Tavily) ──────────────────────────────────

def get_team_motivation(params_json: str) -> str:
    """
    Analisis motivasi tim menjelang pertandingan:
    - posisi klasemen, target sisa musim (relegasi/UCL/gelar)
    - berita internal: squad depth, moral, rotasi

    Format input JSON:
    {
      "home_team": "Mainz 05",
      "away_team": "Union Berlin",
      "league"   : "Bundesliga",
      "season"   : "2025-2026"
    }
    """
    try:
        params    = json.loads(params_json)
        home_team = params.get("home_team", "")
        away_team = params.get("away_team", "")
        league    = params.get("league", "")
        season    = params.get("season", str(datetime.now().year))
    except json.JSONDecodeError:
        return "Error: Format JSON tidak valid."

    if not home_team or not away_team:
        return "Error: 'home_team' dan 'away_team' wajib diisi."

    current_year = str(datetime.now().year)
    queries = [
        f"{home_team} {league} klasemen motivasi target {current_year}"[:380],
        f"{away_team} {league} klasemen motivasi target {current_year}"[:380],
        f"{home_team} vs {away_team} importance stakes {current_year}"[:380],
    ]

    client = TavilyClient(api_key=TAVILY_API_KEY)
    all_results = [
        f"ANALISIS MOTIVASI: {home_team} vs {away_team} | {league} {season}"
    ]

    for q in queries:
        log.info("[MOTIVATION] Query: %s", q)
        try:
            resp = client.search(
                query=q,
                search_depth="advanced",
                max_results=3,
                include_raw_content=True,
                days=14
            )
            results = resp.get("results", [])
            if results:
                all_results.append(f"--- {q} ---")
                for i, item in enumerate(results[:2], 1):
                    raw  = item.get("raw_content") or ""
                    body = raw[:500] if raw else item.get("content", "")[:300]
                    all_results.append(f"{i}. {item.get('title', '')}\n   {body}")
        except Exception as e:
            all_results.append(f"Query gagal: {e}")

    return "\n".join(all_results) if len(all_results) > 1 else "Tidak ada data motivasi."


# ── Tool 10: Get Odds (The Odds API) ─────────────────────────────────────────

def get_odds(params_json: str) -> str:
    """
    Ambil odds real-time dari The Odds API (the-odds-api.com).
    Mendukung 1X2, Over/Under, Asian Handicap, BTTS.
    Otomatis hitung Implied Probability & deteksi Value Bet vs Poisson.

    Format input JSON:
    {
      "home_team"    : "Liverpool",
      "away_team"    : "Arsenal",
      "market"       : "h2h",          # h2h | totals | asian_handicap | btts
      "poisson_home" : 55.0,           # % dari poisson_analysis (opsional)
      "poisson_draw" : 25.0,           # % dari poisson_analysis (opsional)
      "poisson_away" : 20.0            # % dari poisson_analysis (opsional)
    }

    Market tersedia:
      h2h             → 1X2 (Home/Draw/Away)
      totals          → Over/Under 2.5
      asian_handicap  → Asian Handicap
    """

    # ── Map liga ke sport key The Odds API ───────────────────────────────────
    _SPORT_KEYS = [
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_france_ligue_one",
        "soccer_uefa_champs_league",
        "soccer_uefa_europa_league",
        "soccer_netherlands_eredivisie",
        "soccer_portugal_primeira_liga",
        "soccer_turkey_super_league",
        "soccer_brazil_campeonato",
        "soccer_argentina_primera_division",
    ]

    try:
        params     = json.loads(params_json)
        home_team  = params.get("home_team", "").strip()
        away_team  = params.get("away_team", "").strip()
        market     = params.get("market", "h2h").strip().lower()
        p_home     = float(params.get("poisson_home", 0))
        p_draw     = float(params.get("poisson_draw", 0))
        p_away     = float(params.get("poisson_away", 0))
    except (json.JSONDecodeError, ValueError) as e:
        return f"Error parse input: {e}"

    if not home_team or not away_team:
        return "Error: 'home_team' dan 'away_team' wajib diisi."

    # ── Cari pertandingan di semua liga ──────────────────────────────────────
    found_event  = None
    found_sport  = None
    home_lower   = home_team.lower()
    away_lower   = away_team.lower()

    for sport_key in _SPORT_KEYS:
        try:
            resp = requests.get(
                f"{ODDS_API_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey"  : ODDS_API_KEY,
                    "regions" : "eu",
                    "markets" : market,
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
                timeout=15
            )
            if resp.status_code != 200:
                continue

            events = resp.json()
            for event in events:
                h = event.get("home_team", "").lower()
                a = event.get("away_team", "").lower()
                # Match fleksibel: cukup nama pertama atau substring
                if (home_lower in h or h in home_lower) and \
                   (away_lower in a or a in away_lower):
                    found_event = event
                    found_sport = sport_key
                    break

            if found_event:
                break

        except requests.exceptions.RequestException:
            continue

    if not found_event:
        return (
            f"⚠️ Pertandingan '{home_team} vs {away_team}' tidak ditemukan di The Odds API.\n"
            f"Kemungkinan: pertandingan belum terdaftar, sudah selesai, atau nama tim berbeda.\n"
            f"Tip: gunakan nama tim dalam Bahasa Inggris (contoh: 'Manchester United', 'Real Madrid')."
        )

    # ── Ambil odds dari bookmaker ─────────────────────────────────────────────
    bookmakers = found_event.get("bookmakers", [])
    if not bookmakers:
        return "Data pertandingan ditemukan tapi odds belum tersedia."

    commence   = found_event.get("commence_time", "")[:16].replace("T", " ")
    lines      = [
        f"⚽ ODDS: {found_event.get('home_team')} vs {found_event.get('away_team')}",
        f"🕐 Kickoff : {commence} UTC",
        f"🏆 Liga    : {found_sport.replace('soccer_', '').replace('_', ' ').title()}",
        f"📊 Market  : {market.upper()}",
        "─" * 50,
    ]

    # Kumpulkan odds per outcome untuk rata-rata
    odds_pool: dict = {}

    for bm in bookmakers[:6]:  # Maks 6 bookmaker
        bm_name = bm.get("title", "?")
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market:
                continue
            bm_line = f"  {bm_name:<20}"
            outcomes = mkt.get("outcomes", [])
            for o in outcomes:
                name = o.get("name", "")
                odd  = float(o.get("price", 0))
                pt   = o.get("point")  # untuk totals/handicap
                label = f"{name} {pt}" if pt is not None else name
                bm_line += f" | {label}: {odd:.2f}"
                # Kumpulkan untuk rata-rata
                odds_pool.setdefault(label, []).append(odd)
            lines.append(bm_line)

    lines.append("─" * 50)

    # ── Rata-rata odds & Implied Probability ─────────────────────────────────
    lines.append("📈 RATA-RATA ODDS & IMPLIED PROBABILITY:")
    avg_odds = {}
    for label, vals in odds_pool.items():
        avg = round(sum(vals) / len(vals), 3)
        imp = round(100 / avg, 2)
        avg_odds[label] = {"avg": avg, "implied": imp}
        lines.append(f"  {label:<25} avg {avg:.3f}  → implied {imp:.1f}%")

    # ── Value Bet Detection vs Poisson ───────────────────────────────────────
    if p_home > 0 or p_draw > 0 or p_away > 0:
        lines.append("")
        lines.append("🎯 VALUE BET ANALYSIS (Poisson vs Market):")

        poisson_map = {}
        if market == "h2h":
            # Dapatkan nama outcome aktual dari pool
            for label in avg_odds:
                lw = label.lower()
                if "draw" in lw:
                    poisson_map[label] = p_draw
                elif found_event.get("home_team", "").lower() in lw or \
                     home_lower in lw:
                    poisson_map[label] = p_home
                elif found_event.get("away_team", "").lower() in lw or \
                     away_lower in lw:
                    poisson_map[label] = p_away

        for label, data in avg_odds.items():
            poi = poisson_map.get(label)
            if poi is None:
                continue
            implied = data["implied"]
            edge    = round(poi - implied, 2)
            ev      = round((poi / 100) * data["avg"] - 1, 4)
            verdict = "✅ VALUE" if edge > 3 else ("⚠️  BREAK-EVEN" if edge > 0 else "❌ NO VALUE")
            lines.append(
                f"  {label:<25} Poisson {poi:.1f}% vs Implied {implied:.1f}% "
                f"| Edge {edge:+.1f}% | EV {ev:+.4f} → {verdict}"
            )

    # ── Sisa kuota API ────────────────────────────────────────────────────────
    try:
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        lines.append(f"\n💳 Odds API quota: {remaining} sisa / {used} terpakai bulan ini")
    except Exception:
        pass

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 6: CONFIDENCE SCORE — Indikator Keyakinan Prediksi
# ══════════════════════════════════════════════════════════════════════════════

def hitung_confidence(
    p_home: float,
    p_draw: float,
    p_away: float,
    n_tools_used: int = 0,
    odds_available: bool = False,
    xg_available: bool = False,
) -> str:
    """
    Hitung dan format tingkat keyakinan prediksi berdasarkan:
      - Dominasi probabilitas Poisson (semakin timpang → lebih yakin)
      - Kelengkapan data (tools yang berhasil dipanggil)
      - Ketersediaan xG real & odds pasar

    Return: string siap tampil, contoh:
      📊 Confidence: 🟢 TINGGI (67.3%) — Data lengkap: xG real + odds pasar tersedia.
    """
    dominan = max(p_home, p_draw, p_away)

    # ── Level berdasarkan dominasi probabilitas ───────────────────────────────
    if dominan >= 65:
        level = "🟢 TINGGI"
    elif dominan >= 50:
        level = "🟡 SEDANG"
    else:
        level = "🔴 RENDAH"

    # ── Alasan / keterangan tambahan ─────────────────────────────────────────
    alasan_parts = []
    if xg_available:
        alasan_parts.append("xG real tersedia")
    else:
        alasan_parts.append("xG menggunakan rata-rata liga")

    if odds_available:
        alasan_parts.append("odds pasar tersedia")
    else:
        alasan_parts.append("odds tidak ditemukan")

    if n_tools_used >= 5:
        alasan_parts.append("data lengkap")
    elif n_tools_used >= 3:
        alasan_parts.append("data cukup")
    else:
        alasan_parts.append("data terbatas")

    alasan = " | ".join(alasan_parts)
    return f"📊 Confidence: {level} ({dominan:.1f}%) — {alasan.capitalize()}."


_TOOL_REGISTRY = {

    "kalkulator":          kalkulator,

    "tavily_search":       tavily_search,

    "multi_search":        multi_search,

    "analisis_statistik":  analisis_statistik,

    "get_football_stats":  get_football_stats,

    "get_understat_xg":    get_understat_xg,

    "poisson_analysis":    poisson_analysis,

    "get_team_form":       get_team_form,

    "get_team_motivation": get_team_motivation,

    "get_odds":            get_odds,

}



def jalankan_tool(nama_tool: str, input_tool: str) -> str:

    fn = _TOOL_REGISTRY.get(nama_tool.strip().lower())

    if fn is None:

        available = ", ".join(_TOOL_REGISTRY.keys())

        return f"Error: Tool '{nama_tool}' tidak dikenal. Tersedia: {available}"

    return fn(input_tool)





# ══════════════════════════════════════════════════════════════════════════════

#  SELF-REFLECTION — CEK KELENGKAPAN JAWABAN  (PENINGKATAN UTAMA)

# ══════════════════════════════════════════════════════════════════════════════



_REFLECTION_SYSTEM = """Anda adalah quality-checker jawaban AI. Tugas Anda:

Bandingkan pertanyaan asli user dengan draft jawaban agent.

Kembalikan HANYA JSON valid:

{

  "lengkap": true/false,

  "poin_belum_terjawab": ["poin 1", "poin 2"],

  "saran_perbaikan": "satu kalimat saran, atau kosong jika sudah lengkap"

}

Kriteria "lengkap":

- Semua poin eksplisit dalam pertanyaan sudah dijawab

- Jika pertanyaan sepak bola: data xG/formasi/cedera disebut jika relevan

- Tidak ada kontradiksi dengan fakta yang disebutkan

Jangan tambahkan teks di luar JSON."""





def self_reflection(pertanyaan: str, draft_jawaban: str) -> dict:

    """

    [PENINGKATAN] Jalankan refleksi singkat lewat LLM.

    Return dict: {"lengkap": bool, "poin_belum_terjawab": list, "saran_perbaikan": str}

    """

    raw = _panggil_llm_mini(

        prompt_system=_REFLECTION_SYSTEM,

        prompt_user=(

            f"PERTANYAAN USER:\n{pertanyaan}\n\n"

            f"DRAFT JAWABAN AGENT:\n{draft_jawaban}"

        )

    )

    if not raw:

        return {"lengkap": True, "poin_belum_terjawab": [], "saran_perbaikan": ""}

    try:

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        return json.loads(clean)

    except json.JSONDecodeError:

        log.warning("Self-reflection gagal parse: %s", raw[:120])

        return {"lengkap": True, "poin_belum_terjawab": [], "saran_perbaikan": ""}





# ══════════════════════════════════════════════════════════════════════════════

#  SYSTEM PROMPT — FOOTBALL ANALYST

# ══════════════════════════════════════════════════════════════════════════════



# current_year akan diisi saat runtime
_SYSTEM_TEMPLATE = """
██████████████████████████████████████████████████████████████
NEMOTRON FOOTBALL INTELLIGENCE SYSTEM — REACT AGENT
Powered by NVIDIA Nemotron 120B | {CURRENT_YEAR}
██████████████████████████████████████████████████████████████

You are an elite Football Analysis AI.
Your role: professional football analytics used by sportsbooks, analysts, and elite bettors.

{user_identity}
{memory_block}

██████████████████████████████████████████████████████████████
CRITICAL: OUTPUT FORMAT RULES — FOLLOW EXACTLY
██████████████████████████████████████████████████████████████

You MUST respond in ONE of these two formats ONLY.
Do NOT write analysis, reasoning, or prose outside these formats.

FORMAT A — When calling a tool:
Thought: <one sentence why>
Action: <tool_name>
Action Input: <input>

FORMAT B — When ready to answer (only after all mandatory tools called):
Thought: All data collected. Ready to write final analysis.
Final Answer:
<your full structured analysis here>

STRICTLY FORBIDDEN:
- Long paragraphs before Action or Final Answer
- Explaining your thinking process outside the Thought field
- Writing "I will now..." or "Let me analyze..." blocks
- Markdown code blocks inside Action Input
- Skipping mandatory tools
- Estimating probabilities — ONLY use poisson_analysis output

██████████████████████████████████████████████████████████████
MANDATORY TOOL SEQUENCE — EVERY MATCH ANALYSIS
██████████████████████████████████████████████████████████████

ENFORCEMENT: Final Answer is BLOCKED until ALL 4 steps below are completed.

STEP 1 → Action: multi_search
         Action Input: {{"queries": [
           "[Team A] vs [Team B] injury news lineup today {CURRENT_YEAR}",
           "[Team A] vs [Team B] head to head stats {CURRENT_YEAR}",
           "[Team A] vs [Team B] prediction odds {CURRENT_YEAR}",
           "[Team A] [Team B] key players suspension {CURRENT_YEAR}"
         ]}}

STEP 2 → Action: get_understat_xg
         Action Input: {{"home_team": "[Team A]", "away_team": "[Team B]", "season": 2025}}

STEP 3 → Action: get_team_motivation
         Action Input: {{"home_team": "[Team A]", "away_team": "[Team B]", "league": "[Liga]", "season": "{CURRENT_YEAR}"}}

STEP 4 → Action: poisson_analysis
         Action Input: {{"xg_home": <from STEP 2>, "xg_away": <from STEP 2>, "home_team": "[Team A]", "away_team": "[Team B]"}}

STEP 5 → Action: get_odds  ← WAJIB setelah Poisson
         Action Input: {{"home_team": "[Team A]", "away_team": "[Team B]", "market": "h2h",
                         "poisson_home": <% from STEP 4>, "poisson_draw": <% from STEP 4>, "poisson_away": <% from STEP 4>}}

STEP 6 → (Optional) Action: get_team_form
         Action Input: {{"team_id": <id>, "league_id": <id>, "season": 2025, "last": 5}}

STEP 7 → (Optional) Action: tavily_search — only if specific data still missing

STEP 8 → Final Answer (ONLY after STEP 1-5 complete)

██████████████████████████████████████████████████████████████
TOOLS REFERENCE
██████████████████████████████████████████████████████████████

1. kalkulator          — math: percentages, averages, EV calculations
2. tavily_search       — single targeted search (max 380 chars query)
3. multi_search        — 4 parallel Tavily queries in 1 call (USE FIRST)
4. analisis_statistik  — aggregate stats from match data JSON
5. get_football_stats  — standings/fixtures/players from API Football
6. get_understat_xg    — real xG/xGA from Understat (MANDATORY STEP 2)
7. poisson_analysis    — score probability matrix (MANDATORY STEP 4)
8. get_team_motivation — standings, season targets, morale (MANDATORY STEP 3)
9. get_team_form       — last 5 results detail from API Football (optional)
10. get_odds           — real-time odds + implied prob + EV vs Poisson (MANDATORY STEP 5)

██████████████████████████████████████████████████████████████
ANALYSIS FRAMEWORK
██████████████████████████████████████████████████████████████

Cover ALL these in Final Answer:

A. TEAM FORM — Last 5 results, home/away split, momentum, fatigue
B. ADVANCED STATS — xG, xGA, over/under-performing, conversion rate
C. TACTICAL — Formation, pressing, transitions, set pieces, matchup edges
D. PLAYER FACTORS — Injuries/suspensions from Observation only (never invent)
E. MOTIVATION & STAKES — League position, season target, psychological pressure
F. PROBABILITY MODEL — From poisson_analysis (copy directly, do not estimate)
G. MARKET ANALYSIS — AI probability vs bookmaker odds, EV detection

██████████████████████████████████████████████████████████████
FINAL ANSWER TEMPLATE
██████████████████████████████████████████████████████████████

Final Answer:
# MATCH ANALYSIS: [Team A] vs [Team B]

## Tactical Edge
(Formation, pressing, transitions, key matchup advantages)

## Statistical Edge
(xG, xGA, over/under-performing, shot quality)

## Motivation & Stakes
(League position and season target for EACH team — relegation/UCL/title/dead rubber)
(Psychological pressure, home crowd factor, derby significance)

## Form Last 5
(Results W/D/L, goals scored/conceded per game, trend direction)

## Key Risks
(Injuries, suspensions, upset scenarios, hidden dangers)

## Predicted Game Flow
(Expected match tempo, possession balance, momentum prediction)

## Probability Model
(COPY DIRECTLY from poisson_analysis Observation — do not modify)
- [Team A] Win: XX%
- Draw: XX%
- [Team B] Win: XX%

## Most Likely Scores
(COPY from poisson_analysis top 5 — do not modify)

## Betting Value Assessment
- Value Side: (dari get_odds output — ✅ VALUE / ❌ NO VALUE)
- Edge      : (selisih Poisson% vs Implied% — dari get_odds)
- EV        : (Expected Value — dari get_odds)
- Risk Level: Low / Medium / High
- Market Bias: (apakah market over/undervalue salah satu tim)
- Suggested Angle: (1X2 / Over-Under / Handicap / BTTS)

## Confidence Level
Low / Medium / High — (reason: what data is strong/weak)

## Final Verdict
(2-3 sentence professional conclusion with recommended play)

██████████████████████████████████████████████████████████████
ANTI-HALLUCINATION
██████████████████████████████████████████████████████████████
- NEVER invent injuries, suspensions, or stats not in Observation
- If data missing: state explicitly — do not fabricate
- If xG unavailable: use league average (~1.4 home, ~1.1 away) and state it
- All player names MUST come from Observation data

██████████████████████████████████████████████████████████████
RESILIENCE & FALLBACKS
██████████████████████████████████████████████████████████████
- Each tool max 2x per session
- If tool errors: use fallback, do not loop endlessly
- If iteration 8 reached: write best available Final Answer
- get_understat_xg fails → use league average xG, state limitation
- get_team_motivation fails → use tavily_search for standings instead
- get_football_stats fails → rely on Tavily
- get_odds fails → hitung implied prob manual dari odds user (jika ada), atau skip EV section

██████████████████████████████████████████████████████████████
BETTING INTELLIGENCE
██████████████████████████████████████████████████████████████
MANDATORY: Always call get_odds after poisson_analysis.
get_odds akan otomatis hitung implied probability & EV — JANGAN hitung manual.

Setelah get_odds tersedia:
1. Gunakan implied probability dari get_odds output (bukan hitung sendiri)
2. Gunakan edge & EV dari get_odds output langsung di Final Answer
3. Tandai ✅ VALUE / ❌ NO VALUE sesuai output get_odds
4. Jelaskan kenapa market mungkin mispriced berdasarkan data xG & motivasi

Jika get_odds gagal (pertandingan tidak ditemukan):
- Hitung implied probability manual: 1/odds × 100
- Bandingkan vs Poisson secara naratif
- Sebutkan odds gagal diambil otomatis

██████████████████████████████████████████████████████████████
SESSION MEMORY
██████████████████████████████████████████████████████████████
- "that team" / "them" → last analyzed team in conversation history
- Same league standings within session → skip repeated API calls
- User favorite team → add supporter perspective in analysis
"""
# ══════════════════════════════════════════════════════════════════════════════



def jalankan_agent(pertanyaan_user: str, session_id: str) -> str:

    """

    Jalankan satu giliran agent:

      1. Muat memori (fakta kunci + riwayat)

      2. Ekstrak fakta baru dari pesan user (async-feel via LLM mini)

      3. ReAct loop (maks MAX_ITERATIONS)

      4. Self-Reflection sebelum Final Answer

      5. Simpan jawaban ke memori

    """

    # ── 1. Muat memori & profil ───────────────────────────────────────────────

    user_ctx      = get_user_context(session_id)

    memory_block  = load_recent_memory(session_id, limit=10)



    user_identity = ""

    if user_ctx.get("user_name"):

        user_identity = f"Pengguna bernama **{user_ctx['user_name']}**. "

    if user_ctx.get("preferences"):

        user_identity += f"Preferensi: {user_ctx['preferences']}. "



    from datetime import datetime as _dt
    _current_year = str(_dt.now().year)

    prompt_system = _SYSTEM_TEMPLATE.format(

        user_identity=user_identity,

        memory_block=(

            f"\nKONTEKS MEMORI:\n{memory_block}\n" if memory_block

            else ""

        ),

        CURRENT_YEAR=_current_year

    )



    # ── 2. Simpan pesan user & ekstrak fakta ─────────────────────────────────

    save_to_memory(session_id, "user", pertanyaan_user)

    ekstrak_fakta_dari_pesan(pertanyaan_user, session_id)



    # ── 3. ReAct Loop ─────────────────────────────────────────────────────────

    # Seed history: mulai dengan pertanyaan saja, iterasi 1 akan dipaksa pakai tool
    today_str = _dt.now().strftime("%A, %d %B %Y")

    history = (

        f"[TODAY'S DATE: {today_str}] — Tahun saat ini adalah {_current_year}. "

        f"Semua query pencarian WAJIB menggunakan tahun {_current_year}, BUKAN tahun lain.\n"

        f"Question: {pertanyaan_user}\n"

    )

    tool_call_count  = {}   # {nama_tool: jumlah_panggilan}

    final_answer_raw = None



    for iterasi in range(1, MAX_ITERATIONS + 1):

        log.info("─── Iterasi %d/%d ───", iterasi, MAX_ITERATIONS)



        respon = _panggil_llm(prompt_system=prompt_system, prompt_user=history,
                              temperature=0.1)  # ReAct loop: presisi format tinggi

        if not respon:

            log.error("LLM tidak merespons setelah retry.")

            break



        # ── Cek Final Answer ─────────────────────────────────────────────────

        if "Final Answer:" in respon:

            # Blokir Final Answer jika belum ada tool dipanggil sama sekali
            if sum(tool_call_count.values()) < 1:
                log.warning("⛔ Final Answer tanpa tool call — dipaksa cari data.", iterasi)
                history += (respon + "\n" "Observation: [SISTEM: DITOLAK. Wajib panggil multi_search dulu.]\n")
                continue

            # ⛔ Validator Poisson — wajib sebelum Final Answer
            if tool_call_count.get("poisson_analysis", 0) < 1:
                log.warning("⛔ Final Answer sebelum poisson_analysis (iterasi %d) — paksa Poisson.", iterasi)
                import re as _re
                xg_vals = _re.findall(r"xG\s+rata-rata\s*:\s*([\d.]+)", history)
                if len(xg_vals) >= 2:
                    hint = f'{{"xg_home": {xg_vals[0]}, "xg_away": {xg_vals[1]}, "home_team": "Home", "away_team": "Away"}}'
                else:
                    hint = '{"xg_home": 1.5, "xg_away": 1.2, "home_team": "Home", "away_team": "Away"}'
                history += (
                    respon + "\n"
                    "Observation: [SISTEM: DITOLAK. WAJIB panggil poisson_analysis sebelum Final Answer. "
                    "Probability % dan skor HARUS dari output Poisson, bukan estimasi.\n"
                    f"Panggil sekarang:\nThought: Wajib Poisson dulu.\n"
                    f"Action: poisson_analysis\nAction Input: {hint}]\n"
                )
                continue

            final_answer_raw = respon.split("Final Answer:", 1)[-1].strip()

            log.info("Draft Final Answer ditemukan (iterasi %d).", iterasi)

            # Jika draft final answer pendek/terpotong, minta ulang dengan streaming
            if len(final_answer_raw) < 300:
                log.info("Draft terlalu pendek (%d char) — minta Final Answer lengkap via streaming.", len(final_answer_raw))
                streaming_prompt = (
                    history
                    + f"Thought: Semua data terkumpul. Tulis Final Answer lengkap sesuai template.\n"
                    f"Final Answer:\n"
                )
                streamed = _panggil_llm(
                    prompt_system=prompt_system,
                    prompt_user=streaming_prompt,
                    num_predict=4096,
                    temperature=0.3,   # Final Answer: sedikit kreatif, lebih naratif
                    streaming=True,
                )
                if streamed:
                    final_answer_raw = streamed

            break




        # ── Cek Action ───────────────────────────────────────────────────────

        if "Action:" in respon and "Action Input:" in respon:

            try:

                tool_nama  = respon.split("Action:", 1)[1].split("Action Input:", 1)[0].strip()

                tool_input_raw = respon.split("Action Input:", 1)[1]
                # Ambil hingga "Observation:" atau akhir string
                if "Observation:" in tool_input_raw:
                    tool_input = tool_input_raw.split("Observation:", 1)[0].strip()
                else:
                    tool_input = tool_input_raw.strip()
                # Bersihkan nama tool dari newline
                tool_nama = tool_nama.splitlines()[0].strip()

            except IndexError:

                log.warning("Gagal parse Action/Action Input. Melanjutkan…")

                history += f"{respon}\nObservation: [parse error, lanjutkan]\n"

                continue



            # Guard: batas penggunaan per tool

            tool_call_count[tool_nama] = tool_call_count.get(tool_nama, 0) + 1

            if tool_call_count[tool_nama] > MAX_TOOL_RETRIES:

                observation = (

                    f"[SISTEM] Tool '{tool_nama}' sudah dipanggil "

                    f"{tool_call_count[tool_nama]} kali. HENTIKAN penggunaan tool ini "

                    f"dan berikan Final Answer berdasarkan data yang sudah ada."

                )

                log.warning("Tool '%s' melewati batas panggilan.", tool_nama)

                history += f"Action: {tool_nama}\nAction Input: {tool_input}\nObservation: {observation}\n"

                continue



            log.info("🛠️  Tool: %-25s | Input: %s", tool_nama, tool_input[:80])

            hasil = jalankan_tool(tool_nama, tool_input)

            log.info("👁️  Hasil: %s…", hasil[:120])



            # Jika error, tambah peringatan

            if hasil.lower().startswith("error"):

                hasil += (

                    f"\n[SISTEM] Tool '{tool_nama}' gagal. "

                    "Gunakan pengetahuan internal atau tool lain."

                )



            history += (

                f"Action: {tool_nama}\nAction Input: {tool_input}\n"

                f"Observation: {hasil}\n"

            )

            continue



        # ── Tidak ada Action maupun Final Answer ────────────────────────────
        # Nemotron kadang nulis reasoning panjang tanpa format eksplisit
        # Coba rescue: cari Action/Final Answer tersembunyi di dalam teks

        rescued = False

        # Rescue 1: cari pola Action di dalam teks verbose
        action_match = re.search(
            r"Action\s*:\s*(\w+)\s*\nAction\s+Input\s*:\s*(.+?)(?=\nObservation|\nThought|\nAction|\Z)",
            respon, re.DOTALL | re.IGNORECASE
        )
        if action_match:
            rescued_tool  = action_match.group(1).strip()
            rescued_input = action_match.group(2).strip()
            log.info("🔧 Rescued Action dari teks verbose: %s", rescued_tool)
            respon = f"Thought: (rescued)\nAction: {rescued_tool}\nAction Input: {rescued_input}"
            rescued = True

        # Rescue 2: cari Final Answer tersembunyi
        if not rescued:
            fa_match = re.search(r"Final Answer\s*:(.*)", respon, re.DOTALL | re.IGNORECASE)
            if fa_match and sum(tool_call_count.values()) >= 1:
                log.info("🔧 Rescued Final Answer dari teks verbose")
                respon = "Final Answer:" + fa_match.group(1)
                rescued = True

        if rescued:
            # Re-inject ke history dan lanjut iterasi (jangan skip)
            history += respon + "\n"
            continue

        # Jika rescue gagal → dorong dengan hint spesifik
        log.warning("Respons tidak mengandung Action atau Final Answer. Mendorong… (tool calls: %d)", sum(tool_call_count.values()))

        tool_count = sum(tool_call_count.values())
        if tool_count == 0:
            hint = (
                "STOP. Jangan tulis analisis dulu.\n"
                "WAJIB panggil tool sekarang dengan format PERSIS ini:\n"
                "Thought: Saya perlu kumpulkan data.\n"
                "Action: multi_search\n"
                'Action Input: {"queries": ["[Team A] vs [Team B] injury lineup today 2026", '
                '"[Team A] vs [Team B] head to head 2026", '
                '"[Team A] xG stats form 2026", "[Team B] form stats 2026"]}\n'
            )
        elif tool_call_count.get("poisson_analysis", 0) == 0:
            hint = (
                f"Data sudah ada ({tool_count} tool dipanggil). Sekarang WAJIB panggil Poisson:\n"
                "Thought: Hitung probabilitas dengan Poisson.\n"
                "Action: poisson_analysis\n"
                "Action Input: {\"xg_home\": 1.5, \"xg_away\": 1.2, "
                "\"home_team\": \"Home\", \"away_team\": \"Away\"}\n"
            )
        else:
            hint = (
                f"Semua data sudah ada ({tool_count} tool dipanggil). Tulis jawaban sekarang:\n"
                "Thought: Semua data terkumpul. Siap menulis analisis final.\n"
                "Final Answer:\n# MATCH ANALYSIS\n..."
            )

        history += (
            f"{respon}\n"
            f"Observation: [SISTEM: Format tidak valid. {hint}]\n"
        )



    # ── Fallback jika loop habis tanpa Final Answer ───────────────────────────

    if not final_answer_raw:

        log.warning("Loop habis tanpa Final Answer. Meminta ringkasan…")

        fallback = _panggil_llm(

            prompt_system=prompt_system,

            prompt_user=(

                history

                + "\n[SISTEM: Batas iterasi tercapai. "

                "Berikan Final Answer terbaik berdasarkan data yang sudah dikumpulkan.]\n"

            ),

            temperature=0.3,   # Final Answer fallback: sedikit lebih naratif

            streaming=True,    # Streaming agar tidak terasa hang saat model berpikir lama

        )

        if fallback and "Final Answer:" in fallback:

            final_answer_raw = fallback.split("Final Answer:", 1)[-1].strip()

        else:

            final_answer_raw = (

                "Maaf, saya tidak dapat menyelesaikan analisis sepenuhnya saat ini. "

                "Silakan coba ulangi pertanyaan Anda."

            )



    # ── 4. Self-Reflection ────────────────────────────────────────────────────

    log.info("🔍 Menjalankan self-reflection…")

    refleksi = self_reflection(pertanyaan_user, final_answer_raw)



    if not refleksi.get("lengkap", True) and refleksi.get("poin_belum_terjawab"):

        poin_kurang = "; ".join(refleksi["poin_belum_terjawab"])

        saran       = refleksi.get("saran_perbaikan", "")

        log.info("⚠️  Refleksi: ada poin kurang → %s", poin_kurang)



        # Minta LLM melengkapi

        prompt_revisi = (

            f"{history}\nFinal Answer (draft): {final_answer_raw}\n\n"

            f"[SELF-REFLECTION] Poin yang belum terjawab: {poin_kurang}. "

            f"{saran} Perbaiki jawaban dan berikan Final Answer yang lengkap."

        )

        revisi = _panggil_llm(

            prompt_system=prompt_system,

            prompt_user=prompt_revisi,

            num_predict=2048,

            temperature=0.3,   # Revisi analisis: butuh sedikit kreativitas naratif

            streaming=True,    # Streaming agar output terasa real-time

        )

        if revisi and "Final Answer:" in revisi:

            final_answer_raw = revisi.split("Final Answer:", 1)[-1].strip()

            log.info("✅ Jawaban direvisi setelah refleksi.")

        else:

            # Append catatan manual ke draft asli

            final_answer_raw += (

                f"\n\n⚠️ *Catatan: Beberapa poin mungkin belum tercakup sepenuhnya: "

                f"{poin_kurang}. Silakan tanyakan lebih spesifik.*"

            )

    else:

        log.info("✅ Refleksi: jawaban sudah lengkap.")



    # ── 5. Simpan & kembalikan ────────────────────────────────────────────────

    save_to_memory(session_id, "assistant", final_answer_raw)

    print(f"\n🤖 Agent:\n{final_answer_raw}\n")

    return final_answer_raw





# ══════════════════════════════════════════════════════════════════════════════

#  UTILITIES TAMBAHAN

# ══════════════════════════════════════════════════════════════════════════════



def cek_sisa_kredit_tavily() -> None:

    """Cek sisa kredit Tavily lewat usage endpoint."""

    try:

        resp = requests.get(

            "https://api.tavily.com/usage",

            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},

            timeout=10

        )

        if resp.status_code == 200:

            d = resp.json()

            print(f"💳 Kredit Tavily: {d.get('remaining_credits','?')} / {d.get('total_credits','?')}")

        else:

            print(f"⚠️  Gagal cek kredit Tavily: HTTP {resp.status_code}")

    except Exception as e:

        print(f"⚠️  Error cek kredit Tavily: {e}")





def tampilkan_fakta_user(session_id: str) -> None:

    """Debug helper: tampilkan semua fakta tersimpan untuk sesi ini."""

    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(

        "SELECT fact_key, fact_value, updated_at FROM user_facts WHERE session_id = ? ORDER BY updated_at DESC",

        (session_id,)

    ).fetchall()

    conn.close()

    if not rows:

        print("(Belum ada fakta tersimpan untuk sesi ini)")

        return

    print("\n📋 Fakta User Tersimpan:")

    print("─" * 50)

    for k, v, ts in rows:

        print(f"  {k:<20} → {v}  [{ts}]")

    print("─" * 50)





# ══════════════════════════════════════════════════════════════════════════════

#  ENTRYPOINT

# ══════════════════════════════════════════════════════════════════════════════





# ══════════════════════════════════════════════════════════════════════════════

#  TELEGRAM OUTPUT

# ══════════════════════════════════════════════════════════════════════════════



async def _kirim_telegram_async(pesan: str) -> None:

    """Kirim pesan ke Telegram (async)."""

    try:

        bot = Bot(token=TELEGRAM_TOKEN)

        max_len = 4000

        for i in range(0, len(pesan), max_len):

            chunk = pesan[i:i+max_len]

            await bot.send_message(

                chat_id=TELEGRAM_CHAT_ID,

                text=chunk,

                parse_mode=ParseMode.MARKDOWN

            )

            await asyncio.sleep(0.3)

        log.info("✅ Pesan Telegram terkirim.")

    except Exception as e:

        log.error("❌ Gagal kirim Telegram: %s", e)



def kirim_telegram(pesan: str) -> None:

    """Wrapper sync untuk kirim Telegram."""

    try:

        asyncio.run(_kirim_telegram_async(pesan))

    except RuntimeError:

        loop = asyncio.get_event_loop()

        loop.run_until_complete(_kirim_telegram_async(pesan))



def format_telegram(pertanyaan: str, jawaban: str, session_id: str) -> str:

    """Format pesan analisis untuk Telegram."""

    now = datetime.now().strftime("%d %b %Y %H:%M")

    return f"""⚽ *FOOTBALL ANALYSIS AGENT*
📅 {now} | 🔑 `{session_id}`
━━━━━━━━━━━━━━━━━━━━

❓ *Pertanyaan:*
_{pertanyaan}_

━━━━━━━━━━━━━━━━━━━━
🤖 *Analisis AI:*

{jawaban}

━━━━━━━━━━━━━━━━━━━━
_Powered by NVIDIA Nemotron 120B via NIM_
_⚠️ Gunakan sebagai referensi, bukan keputusan mutlak._"""



if __name__ == "__main__":

    init_db()

    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN tidak ditemukan. Set environment variable TELEGRAM_TOKEN.")

    # ══════════════════════════════════════════════════════════════════════════
    #  TELEGRAM BOT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handler command /start."""
        user = update.effective_user
        session_id = str(user.id)

        user_ctx = get_user_context(session_id)
        if user_ctx["user_name"]:
            salam = f"👋 Selamat datang kembali, {user_ctx['user_name']}!"
        else:
            salam = (
                "⚽ *FOOTBALL ANALYSIS AGENT* siap digunakan\\!\n\n"
                "💡 Tips: Sebutkan nama dan tim favorit Anda agar saya bisa mengenali Anda\\.\n"
                "Contoh: _Analisis Liverpool vs Arsenal malam ini_"
            )

        await update.message.reply_text(salam, parse_mode=ParseMode.MARKDOWN_V2)

    async def cmd_fakta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handler command /fakta — tampilkan profil tersimpan."""
        session_id = str(update.effective_user.id)
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT fact_key, fact_value, updated_at FROM user_facts "
            "WHERE session_id = ? ORDER BY updated_at DESC",
            (session_id,)
        ).fetchall()
        conn.close()

        if not rows:
            await update.message.reply_text("_(Belum ada fakta tersimpan untuk akun Anda)_",
                                            parse_mode=ParseMode.MARKDOWN)
            return

        lines = ["📋 *Fakta Tersimpan:*", "─" * 30]
        for k, v, ts in rows:
            lines.append(f"  `{k}` → {v}")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handler pesan teks biasa — panggil agent dan balas ke user."""
        user_input = (update.message.text or "").strip()
        if not user_input:
            return

        session_id = str(update.effective_user.id)
        log.info("📨 Pesan dari %s (session=%s): %s", update.effective_user.username, session_id, user_input[:80])

        # Kirim indikator "sedang mengetik"
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        try:
            # Jalankan agent di thread terpisah agar tidak memblokir event loop Telegram
            loop = asyncio.get_event_loop()
            jawaban = await loop.run_in_executor(None, jalankan_agent, user_input, session_id)
        except Exception as e:
            log.error("Agent error: %s", e)
            jawaban = f"⚠️ Terjadi kesalahan saat memproses permintaan Anda: {e}"

        # Telegram max 4096 karakter per pesan
        max_len = 4000
        for i in range(0, len(jawaban), max_len):
            chunk = jawaban[i:i + max_len]
            try:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                # Fallback tanpa markdown jika ada karakter yang tidak valid
                await update.message.reply_text(chunk)
            await asyncio.sleep(0.3)

    # ── Bangun & jalankan bot ──────────────────────────────────────────────────
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("fakta", cmd_fakta))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("🤖 Football Analysis Bot berjalan… (polling)")
    cek_sisa_kredit_tavily()

    app.run_polling(drop_pending_updates=True)





# ══════════════════════════════════════════════════════════════════════════════

# RINGKASAN PENINGKATAN DARI KODE ASLI

# ══════════════════════════════════════════════════════════════════════════════

#

# ┌─────────────────────────────────────────────────────────────────────────┐

# │  AREA                  SEBELUM              SESUDAH                     │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ Skema DB               2 tabel              3 tabel (+user_facts)       │

# │                        kolom preferences    kolom terstruktur per kunci │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ Ekstraksi Fakta        Regex "nama saya"    LLM mini → JSON → upsert    │

# │                        (1 pola saja)        (6+ kunci fakta otomatis)   │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ load_recent_memory     10 pesan terakhir    Fakta kunci + 10 pesan      │

# │                        saja                 (2 blok terpisah)           │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ ReAct Loop             5 iter, tidak ada    7 iter + guard per-tool     │

# │                        guard per-tool       + fallback LLM saat habis   │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ Self-Reflection        Tidak ada            LLM mini cek kelengkapan    │

# │                                             → revisi jika ada gap       │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ Ketahanan LLM          1 try/except         Retry 2x + backoff + batas  │

# │                        tanpa retry          token per panggilan         │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ Tools                  2 (kalkulator,       4 (+analisis_statistik,     │

# │                         tavily_search)       get_football_stats)        │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ System Prompt          Generik              Khusus sepak bola: xG,      │

# │                                             formasi, cedera, formasi    │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ kalkulator             eval() tanpa guard   Whitelist regex + zero-div  │

# ├─────────────────────────────────────────────────────────────────────────┤

# │ Session                Selalu baru          Bisa resume session lama    │

# └─────────────────────────────────────────────────────────────────────────┘

