"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AGENTIC AI CODING — Football Agent Optimizer                       ║
║          Target: main.py (Football Analysis Agent v2.0)                     ║
║          Engine: NVIDIA Nemotron 120B (ReAct loop)                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TUGAS AGEN INI:                                                             ║
║  1. ANALYZE  — Scan main.py untuk kelemahan (statik + performa)             ║
║  2. DIAGNOSE — Identifikasi prioritas perbaikan via LLM                     ║
║  3. GENERATE — Tulis patch/improvement secara otomatis                      ║
║  4. VALIDATE — Cek sintaks Python sebelum diterapkan                        ║
║  5. APPLY    — Terapkan patch dengan backup otomatis                        ║
║  6. LOG      — Catat semua perubahan ke changelog + SQLite                  ║
║  7. TRACK    — Lacak akurasi prediksi parlay dari waktu ke waktu            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KELEMAHAN YANG DITEMUKAN DI main.py:                                       ║
║  [BUG-1] _normalize_year_in_query → regex 2[0-5] hardcoded (miss 2026+)    ║
║  [BUG-2] Understat season 2025 hardcoded di system prompt                   ║
║  [GAP-1] Tidak ada tabel predictions → akurasi parlay tidak terukur         ║
║  [GAP-2] Poisson tidak pakai home advantage coefficient (+8%)               ║
║  [GAP-3] Confidence scoring hanya threshold sederhana (65/50)               ║
║  [GAP-4] Cache (_CACHE dict) tumbuh tanpa batas → memory leak               ║
║  [GAP-5] get_team_form tidak punya fallback saat FOOTBALL_API_KEY kosong    ║
║  [GAP-6] _REFLECTION_SYSTEM tidak cek parlay-specific info                 ║
║  [GAP-7] Tool performance tidak dicatat (gagal/sukses per tool)             ║
║  [GAP-8] multi_search query template terlalu generik                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import ast
import sys
import json
import time
import shutil
import sqlite3
import logging
import textwrap
import requests
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME      = "nvidia/nemotron-3-super-120b-a12b"
MODEL_MINI      = "nvidia/llama-3.1-nemotron-nano-8b-v1"

TARGET_FILE     = os.environ.get("TARGET_FILE", "main.py")
BACKUP_DIR      = os.environ.get("BACKUP_DIR", "./backups")
DB_PATH         = os.environ.get("CODING_DB_PATH", "/tmp/coding_agent.db")
CHANGELOG_FILE  = "CODING_AGENT_CHANGELOG.md"

MAX_ITERATIONS  = 8    # Iterasi ReAct coding loop
LLM_TIMEOUT     = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("CodingAgent")


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def init_coding_db() -> None:
    """Buat tabel untuk coding agent dan prediction tracker."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Riwayat improvement yang pernah diterapkan
    c.execute('''
        CREATE TABLE IF NOT EXISTS improvements (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            issue_id    TEXT,
            issue_desc  TEXT,
            patch_before TEXT,
            patch_after  TEXT,
            status      TEXT DEFAULT 'proposed',  -- proposed | applied | rejected
            applied_at  DATETIME,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Lacak akurasi prediksi parlay
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            match_home      TEXT,
            match_away      TEXT,
            league          TEXT,
            predicted_pick  TEXT,    -- "home" | "draw" | "away"
            confidence_pct  REAL,
            poisson_home    REAL,
            poisson_draw    REAL,
            poisson_away    REAL,
            xg_home         REAL,
            xg_away         REAL,
            actual_result   TEXT,    -- "home" | "draw" | "away" | NULL (belum diisi)
            is_correct      INTEGER, -- 1 = benar, 0 = salah, NULL = belum diisi
            parlay_id       TEXT,    -- grouping untuk parlay
            predicted_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at     DATETIME
        )
    ''')

    # Log performa tool (tools mana yang sering gagal)
    c.execute('''
        CREATE TABLE IF NOT EXISTS tool_performance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name   TEXT,
            success     INTEGER,  -- 1 atau 0
            duration_ms INTEGER,
            error_msg   TEXT,
            logged_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Sesi coding agent
    c.execute('''
        CREATE TABLE IF NOT EXISTS coding_sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT UNIQUE,
            issues_found INTEGER DEFAULT 0,
            patches_applied INTEGER DEFAULT 0,
            started_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            finished_at DATETIME
        )
    ''')

    conn.commit()
    conn.close()
    log.info("Coding DB diinisialisasi.")


# ══════════════════════════════════════════════════════════════════════════════
#  LLM CALLER
# ══════════════════════════════════════════════════════════════════════════════

def _llm(system: str, user: str, model: str = MODEL_NAME,
         max_tokens: int = 4096, temperature: float = 0.1) -> Optional[str]:
    """Panggil NVIDIA Nemotron dengan retry 2x."""
    if not NVIDIA_API_KEY:
        log.error("NVIDIA_API_KEY tidak ditemukan.")
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       0.7,
        "stream":      False,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type":  "application/json"
    }
    for attempt in range(1, 3):
        try:
            r = requests.post(NVIDIA_CHAT_URL, json=payload,
                              headers=headers, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            # Bersihkan <think> block Nemotron
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return raw
        except Exception as e:
            log.warning("LLM error (attempt %d): %s", attempt, e)
            if attempt < 2:
                time.sleep(3)
    return None

def _llm_mini(system: str, user: str) -> Optional[str]:
    return _llm(system, user, model=MODEL_MINI, max_tokens=1024, temperature=0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  CODING TOOLS — ReAct Loop
# ══════════════════════════════════════════════════════════════════════════════

def tool_read_file(params_json: str) -> str:
    """
    Baca isi file target (main.py) atau sebagian darinya.
    Input JSON: {"file": "main.py", "start_line": 1, "end_line": 100}
    """
    try:
        p = json.loads(params_json)
    except Exception:
        p = {}
    filepath  = p.get("file", TARGET_FILE)
    start     = p.get("start_line", 1)
    end       = p.get("end_line",   None)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if end:
            chunk = lines[start-1:end]
        else:
            chunk = lines[:100]  # Default 100 baris pertama
        numbered = [f"{start+i:>4} | {l.rstrip()}" for i, l in enumerate(chunk)]
        return "\n".join(numbered)
    except FileNotFoundError:
        return f"Error: File '{filepath}' tidak ditemukan."
    except Exception as e:
        return f"Error membaca file: {e}"


def tool_search_in_file(params_json: str) -> str:
    """
    Cari fungsi atau pattern tertentu di main.py.
    Input JSON: {"pattern": "def poisson_analysis", "context_lines": 5}
    """
    try:
        p = json.loads(params_json)
    except Exception:
        p = {"pattern": params_json}
    pattern       = p.get("pattern", "")
    context_lines = p.get("context_lines", 10)

    try:
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"Error: {TARGET_FILE} tidak ditemukan."

    hasil = []
    for i, line in enumerate(lines):
        if pattern.lower() in line.lower():
            start = max(0, i - 2)
            end   = min(len(lines), i + context_lines)
            snippet = [f"{start+j+1:>4} | {lines[start+j].rstrip()}"
                       for j in range(end - start)]
            hasil.append(f"[Match di baris {i+1}]\n" + "\n".join(snippet))
    if not hasil:
        return f"Pattern '{pattern}' tidak ditemukan di {TARGET_FILE}."
    return "\n\n---\n".join(hasil[:5])  # Maks 5 match


def tool_validate_python(params_json: str) -> str:
    """
    Validasi sintaks Python untuk kode yang akan dipatch.
    Input JSON: {"code": "def foo():\\n    return 1"}
    """
    try:
        p = json.loads(params_json)
        code = p.get("code", "")
    except Exception:
        code = params_json

    if not code.strip():
        return "Error: Kode kosong."
    try:
        ast.parse(code)
        return "✅ Sintaks valid — aman untuk diterapkan."
    except SyntaxError as e:
        return f"❌ SyntaxError baris {e.lineno}: {e.msg}"
    except Exception as e:
        return f"❌ Error validasi: {e}"


def tool_apply_patch(params_json: str) -> str:
    """
    Terapkan patch ke main.py dengan backup otomatis.
    Input JSON: {
        "issue_id": "BUG-1",
        "description": "Fix regex normalisasi tahun",
        "old_code": "return re.sub(r'\\\\b20(2[0-5])\\\\b'...",
        "new_code": "current_year = ...",
        "session_id": "sess_001"
    }
    """
    try:
        p = json.loads(params_json)
    except Exception:
        return "Error: Format JSON tidak valid."

    issue_id    = p.get("issue_id",    "UNKNOWN")
    description = p.get("description", "")
    old_code    = p.get("old_code",    "")
    new_code    = p.get("new_code",    "")
    session_id  = p.get("session_id",  "default")

    if not old_code or not new_code:
        return "Error: 'old_code' dan 'new_code' wajib diisi."

    # ── 1. Backup dulu ────────────────────────────────────────────────────────
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"main_{ts}_{issue_id}.py.bak")
    try:
        shutil.copy2(TARGET_FILE, backup_path)
        log.info("Backup dibuat: %s", backup_path)
    except Exception as e:
        return f"Error membuat backup: {e}. Patch DIBATALKAN."

    # ── 2. Validasi sintaks new_code ──────────────────────────────────────────
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        return f"❌ Patch DITOLAK — SyntaxError di new_code: baris {e.lineno}: {e.msg}"

    # ── 3. Terapkan patch (string replace) ───────────────────────────────────
    try:
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        if old_code not in content:
            return (
                f"❌ Patch GAGAL — 'old_code' tidak ditemukan di {TARGET_FILE}.\n"
                f"Pastikan old_code persis sama termasuk spasi & indentasi."
            )

        new_content = content.replace(old_code, new_code, 1)  # Replace SEKALI saja

        with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.write(new_content)

    except Exception as e:
        # Rollback
        shutil.copy2(backup_path, TARGET_FILE)
        return f"❌ Error saat menulis patch: {e}. File di-rollback dari backup."

    # ── 4. Final validation — baca ulang & parse ─────────────────────────────
    try:
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            final_content = f.read()
        ast.parse(final_content)
    except SyntaxError as e:
        shutil.copy2(backup_path, TARGET_FILE)
        return (
            f"❌ File hasil patch TIDAK VALID (SyntaxError baris {e.lineno}).\n"
            f"File di-rollback ke backup: {backup_path}"
        )

    # ── 5. Simpan ke DB ───────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO improvements
               (session_id, issue_id, issue_desc, patch_before, patch_after, status, applied_at)
               VALUES (?, ?, ?, ?, ?, 'applied', ?)""",
            (session_id, issue_id, description, old_code[:500], new_code[:500], datetime.now())
        )
        conn.execute(
            """UPDATE coding_sessions
               SET patches_applied = patches_applied + 1
               WHERE session_id = ?""",
            (session_id,)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("Gagal simpan ke DB: %s", e)

    # ── 6. Tulis changelog ────────────────────────────────────────────────────
    _tulis_changelog(issue_id, description, old_code, new_code, ts)

    return (
        f"✅ Patch [{issue_id}] BERHASIL diterapkan!\n"
        f"   Deskripsi : {description}\n"
        f"   Backup    : {backup_path}\n"
        f"   Changelog : {CHANGELOG_FILE}"
    )


def tool_add_prediction(params_json: str) -> str:
    """
    Catat satu prediksi ke tabel predictions untuk tracking akurasi parlay.
    Input JSON: {
        "session_id": "user123",
        "match_home": "Liverpool", "match_away": "Arsenal", "league": "EPL",
        "predicted_pick": "home",
        "confidence_pct": 62.5,
        "poisson_home": 52.3, "poisson_draw": 27.1, "poisson_away": 20.6,
        "xg_home": 1.72, "xg_away": 1.15,
        "parlay_id": "parlay_20260516"
    }
    """
    try:
        p = json.loads(params_json)
    except Exception:
        return "Error: Format JSON tidak valid."

    required = ["session_id", "match_home", "match_away", "predicted_pick"]
    missing  = [k for k in required if not p.get(k)]
    if missing:
        return f"Error: Field wajib kosong: {missing}"

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO predictions
               (session_id, match_home, match_away, league, predicted_pick,
                confidence_pct, poisson_home, poisson_draw, poisson_away,
                xg_home, xg_away, parlay_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p["session_id"], p["match_home"], p["match_away"],
                p.get("league", ""), p["predicted_pick"],
                p.get("confidence_pct"), p.get("poisson_home"),
                p.get("poisson_draw"), p.get("poisson_away"),
                p.get("xg_home"), p.get("xg_away"),
                p.get("parlay_id", "")
            )
        )
        pred_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()
        return f"✅ Prediksi #{pred_id} dicatat: {p['match_home']} vs {p['match_away']} → {p['predicted_pick']}"
    except Exception as e:
        return f"Error menyimpan prediksi: {e}"


def tool_resolve_prediction(params_json: str) -> str:
    """
    Update hasil aktual pertandingan dan hitung akurasi.
    Input JSON: {
        "prediction_id": 5,
        "actual_result": "home"   -- "home" | "draw" | "away"
    }
    ATAU update by match:
    {
        "match_home": "Liverpool", "match_away": "Arsenal",
        "actual_result": "home"
    }
    """
    try:
        p = json.loads(params_json)
    except Exception:
        return "Error: Format JSON tidak valid."

    actual = p.get("actual_result", "").lower()
    if actual not in ("home", "draw", "away"):
        return "Error: 'actual_result' harus 'home', 'draw', atau 'away'."

    try:
        conn = sqlite3.connect(DB_PATH)

        if p.get("prediction_id"):
            row = conn.execute(
                "SELECT predicted_pick FROM predictions WHERE id = ?",
                (p["prediction_id"],)
            ).fetchone()
            if not row:
                conn.close()
                return f"Error: Prediksi #{p['prediction_id']} tidak ditemukan."
            predicted = row[0]
            is_correct = 1 if predicted == actual else 0
            conn.execute(
                """UPDATE predictions SET actual_result=?, is_correct=?, resolved_at=?
                   WHERE id=?""",
                (actual, is_correct, datetime.now(), p["prediction_id"])
            )
        else:
            home = p.get("match_home", "")
            away = p.get("match_away", "")
            rows = conn.execute(
                """SELECT id, predicted_pick FROM predictions
                   WHERE match_home LIKE ? AND match_away LIKE ?
                   AND actual_result IS NULL""",
                (f"%{home}%", f"%{away}%")
            ).fetchall()
            if not rows:
                conn.close()
                return f"Error: Prediksi '{home} vs {away}' (belum resolve) tidak ditemukan."
            for pid, predicted in rows:
                is_correct = 1 if predicted == actual else 0
                conn.execute(
                    """UPDATE predictions SET actual_result=?, is_correct=?, resolved_at=?
                       WHERE id=?""",
                    (actual, is_correct, datetime.now(), pid)
                )

        conn.commit()
        conn.close()

        # Hitung akurasi terkini
        stats = tool_prediction_stats("{}")
        return f"✅ Hasil diupdate: actual={actual}\n\n{stats}"
    except Exception as e:
        return f"Error update prediksi: {e}"


def tool_prediction_stats(params_json: str) -> str:
    """
    Tampilkan statistik akurasi prediksi parlay.
    Input JSON: {"session_id": "user123"} atau {} untuk semua
    """
    try:
        p = json.loads(params_json) if params_json.strip() else {}
    except Exception:
        p = {}

    session_filter = p.get("session_id", "")

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        where = "WHERE is_correct IS NOT NULL"
        args  = []
        if session_filter:
            where += " AND session_id = ?"
            args.append(session_filter)

        # Total & akurasi
        total = c.execute(
            f"SELECT COUNT(*) FROM predictions {where}", args
        ).fetchone()[0]

        benar = c.execute(
            f"SELECT COUNT(*) FROM predictions {where} AND is_correct=1", args
        ).fetchone()[0]

        # Per-pick accuracy
        for pick in ("home", "draw", "away"):
            pass  # Dihitung di bawah

        # Rata-rata confidence saat benar vs salah
        avg_conf_benar = c.execute(
            f"SELECT AVG(confidence_pct) FROM predictions {where} AND is_correct=1", args
        ).fetchone()[0] or 0

        avg_conf_salah = c.execute(
            f"SELECT AVG(confidence_pct) FROM predictions {where} AND is_correct=0", args
        ).fetchone()[0] or 0

        # Prediksi terbaru (5 terakhir)
        recents = c.execute(
            f"""SELECT match_home, match_away, predicted_pick, actual_result, is_correct, confidence_pct
                FROM predictions {where}
                ORDER BY resolved_at DESC LIMIT 5""", args
        ).fetchall()

        # Parlay win rate (semua prediksi di parlay_id yang sama harus benar)
        parlay_ids = c.execute(
            f"""SELECT DISTINCT parlay_id FROM predictions
                WHERE parlay_id != '' AND is_correct IS NOT NULL""", []
        ).fetchall()

        parlay_win = parlay_total = 0
        for (pid,) in parlay_ids:
            rows = c.execute(
                "SELECT is_correct FROM predictions WHERE parlay_id=?", (pid,)
            ).fetchall()
            if all(r[0] == 1 for r in rows):
                parlay_win += 1
            parlay_total += 1

        conn.close()

        akurasi = (benar / total * 100) if total > 0 else 0
        parlay_rate = (parlay_win / parlay_total * 100) if parlay_total > 0 else 0

        lines = [
            "📊 STATISTIK AKURASI PREDIKSI",
            "─" * 45,
            f"Total Prediksi Resolved : {total}",
            f"Benar                   : {benar}",
            f"Akurasi Single          : {akurasi:.1f}%",
            f"",
            f"🎰 Parlay Win Rate      : {parlay_win}/{parlay_total} = {parlay_rate:.1f}%",
            f"",
            f"📈 Avg Confidence (Benar): {avg_conf_benar:.1f}%",
            f"📉 Avg Confidence (Salah): {avg_conf_salah:.1f}%",
            f"",
            "5 Prediksi Terakhir:",
        ]
        for home, away, pick, actual, correct, conf in recents:
            icon = "✅" if correct else "❌"
            lines.append(
                f"  {icon} {home} vs {away}: pick={pick}, actual={actual}, conf={conf:.0f}%"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error membaca statistik: {e}"


def tool_analyze_code(params_json: str) -> str:
    """
    Gunakan LLM untuk menganalisis fungsi tertentu dan cari masalah.
    Input JSON: {"function_name": "hitung_confidence", "focus": "calibration accuracy"}
    """
    try:
        p = json.loads(params_json)
    except Exception:
        p = {"function_name": params_json}

    func_name = p.get("function_name", "")
    focus     = p.get("focus", "kelemahan umum dan perbaikan")

    # Ambil kode fungsi dari file
    code_snippet = tool_search_in_file(json.dumps({
        "pattern": f"def {func_name}",
        "context_lines": 50
    }))

    if "tidak ditemukan" in code_snippet.lower():
        return f"Fungsi '{func_name}' tidak ditemukan di {TARGET_FILE}."

    analysis = _llm_mini(
        system="""Anda adalah senior Python developer & AI engineer.
Analisis kode Python berikut dan temukan:
1. Bug atau kesalahan logika
2. Kelemahan performa atau akurasi
3. Potensi edge case yang tidak ditangani
4. Rekomendasi perbaikan spesifik (dengan contoh kode)

Jawab dalam Bahasa Indonesia. Format jawaban:
MASALAH: [deskripsi singkat]
DAMPAK: [dampak ke prediksi parlay]
SOLUSI: [kode perbaikan atau saran konkret]""",
        user=f"Fokus analisis: {focus}\n\nKode:\n{code_snippet}"
    )
    return analysis or "Error: LLM tidak merespons."


def tool_generate_patch(params_json: str) -> str:
    """
    Minta LLM menulis patch untuk perbaikan tertentu.
    Input JSON: {
        "issue_id": "BUG-1",
        "issue_description": "Regex tahun hardcoded 2[0-5]",
        "current_code": "def _normalize_year...",
        "improvement_goal": "Jadikan dinamis berdasarkan tahun sekarang"
    }
    """
    try:
        p = json.loads(params_json)
    except Exception:
        return "Error: Format JSON tidak valid."

    issue_id    = p.get("issue_id", "")
    description = p.get("issue_description", "")
    current     = p.get("current_code", "")
    goal        = p.get("improvement_goal", "")

    if not current:
        return "Error: 'current_code' wajib diisi."

    patch = _llm(
        system="""Anda adalah senior Python developer.
Tugas: tulis versi PERBAIKAN dari kode Python yang diberikan.

ATURAN WAJIB:
- Output HANYA kode Python bersih (tanpa markdown backtick, tanpa penjelasan)
- Pertahankan nama fungsi yang sama
- Pertahankan docstring jika ada
- Kode harus valid Python 3.9+
- Jangan tambahkan import baru kecuali sangat diperlukan
- Perbaikan harus spesifik sesuai 'improvement_goal'""",
        user=(
            f"Issue ID   : {issue_id}\n"
            f"Deskripsi  : {description}\n"
            f"Target     : {goal}\n\n"
            f"Kode sekarang:\n{current}"
        ),
        temperature=0.15
    )
    if not patch:
        return "Error: LLM tidak menghasilkan patch."

    # Bersihkan kalau ada markdown
    patch = re.sub(r"```(?:python)?|```", "", patch).strip()

    return f"=== PATCH GENERATED [{issue_id}] ===\n{patch}\n=== END PATCH ==="


def tool_run_static_analysis(params_json: str) -> str:
    """
    Jalankan analisis statis otomatis pada main.py.
    Mendeteksi 10 kelemahan spesifik yang sudah diidentifikasi.
    Input JSON: {} (tidak perlu parameter)
    """
    log.info("Menjalankan static analysis...")
    results = []

    try:
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            lines   = content.split("\n")
    except FileNotFoundError:
        return f"Error: {TARGET_FILE} tidak ditemukan."

    current_year = str(datetime.now().year)
    last_2_digits = current_year[2:]

    # ── BUG-1: Regex tahun hardcoded ──────────────────────────────────────────
    if "2[0-5]" in content:
        results.append({
            "id": "BUG-1", "severity": "HIGH",
            "title": "Regex normalisasi tahun hardcoded",
            "detail": (
                f"_normalize_year_in_query menggunakan '2[0-5]' → hanya cover 2020-2025.\n"
                f"Tahun sekarang {current_year} — query bisa salah tahun!\n"
                f"FIX: Jadikan dinamis: current_year = datetime.now().year"
            )
        })

    # ── BUG-2: Season hardcoded di system prompt ──────────────────────────────
    hardcoded_seasons = re.findall(r'"season":\s*20(2[0-4])', content)
    if hardcoded_seasons:
        results.append({
            "id": "BUG-2", "severity": "HIGH",
            "title": "Season hardcoded di system prompt/kode",
            "detail": (
                f"Ditemukan 'season': 202{hardcoded_seasons[0]} hardcoded di kode.\n"
                f"Harus otomatis: datetime.now().year\n"
                f"FIX: Ganti dengan variabel _current_year yang sudah ada."
            )
        })

    # ── GAP-1: Tidak ada tabel predictions ───────────────────────────────────
    if "predictions" not in content:
        results.append({
            "id": "GAP-1", "severity": "HIGH",
            "title": "Tidak ada tracking akurasi prediksi",
            "detail": (
                "main.py tidak menyimpan hasil prediksi ke DB.\n"
                "Tidak bisa mengukur win rate parlay dari waktu ke waktu.\n"
                "FIX: Tambah tabel 'predictions' dan fungsi record_prediction()."
            )
        })

    # ── GAP-2: Poisson tanpa home advantage ──────────────────────────────────
    poisson_fn = re.search(r"def poisson_analysis.*?(?=def |\Z)", content, re.DOTALL)
    if poisson_fn and "home_advantage" not in poisson_fn.group():
        results.append({
            "id": "GAP-2", "severity": "MEDIUM",
            "title": "Poisson tidak pakai home advantage coefficient",
            "detail": (
                "Distribusi Poisson murni tidak memodelkan keunggulan bermain di kandang.\n"
                "Historical data: home team menang ~49% vs away team ~27%.\n"
                "FIX: Tambahkan home_advantage_factor = 1.08-1.12 pada xg_home sebelum Poisson."
            )
        })

    # ── GAP-3: Confidence scoring terlalu simpel ──────────────────────────────
    conf_fn = re.search(r"def hitung_confidence.*?(?=def |\Z)", content, re.DOTALL)
    if conf_fn:
        fn_body = conf_fn.group()
        if "calibrat" not in fn_body and "historical" not in fn_body:
            results.append({
                "id": "GAP-3", "severity": "MEDIUM",
                "title": "Confidence scoring tidak terkalibrasi dengan data historis",
                "detail": (
                    "hitung_confidence() hanya menggunakan threshold statis (65/50%).\n"
                    "Tidak ada kalibrasi berdasarkan akurasi historis prediksi sebelumnya.\n"
                    "FIX: Query tabel predictions untuk hitung calibrated confidence."
                )
            })

    # ── GAP-4: Cache memory leak ──────────────────────────────────────────────
    if "_CACHE: dict = {}" in content and "maxsize" not in content:
        results.append({
            "id": "GAP-4", "severity": "MEDIUM",
            "title": "_CACHE dict tumbuh tanpa batas (potential memory leak)",
            "detail": (
                "_CACHE tidak punya maxsize limit.\n"
                "Jika bot berjalan lama, cache bisa memakan RAM.\n"
                "FIX: Tambah pembersihan cache saat len(_CACHE) > 1000, atau pakai functools.lru_cache."
            )
        })

    # ── GAP-5: get_team_form tanpa fallback ──────────────────────────────────
    form_fn = re.search(r"def get_team_form.*?(?=def |\Z)", content, re.DOTALL)
    if form_fn and "FOOTBALL_API_KEY" in form_fn.group():
        fn_body = form_fn.group()
        if "tavily" not in fn_body.lower() and "fallback" not in fn_body.lower():
            results.append({
                "id": "GAP-5", "severity": "LOW",
                "title": "get_team_form tidak punya fallback jika FOOTBALL_API_KEY kosong",
                "detail": (
                    "Saat ini, jika FOOTBALL_API_KEY tidak diisi, fungsi langsung return error.\n"
                    "FIX: Fallback ke tavily_search untuk mencari form tim dari web."
                )
            })

    # ── GAP-6: Reflection tidak cek parlay-specific ───────────────────────────
    if "_REFLECTION_SYSTEM" in content:
        refl_match = re.search(r'_REFLECTION_SYSTEM\s*=\s*"""(.*?)"""', content, re.DOTALL)
        if refl_match and "parlay" not in refl_match.group(1).lower():
            results.append({
                "id": "GAP-6", "severity": "LOW",
                "title": "Self-reflection tidak memeriksa kelengkapan info parlay",
                "detail": (
                    "_REFLECTION_SYSTEM tidak memeriksa apakah confidence, value bet,\n"
                    "dan rekomendasi parlay sudah ada di jawaban.\n"
                    "FIX: Tambahkan kriteria parlay ke _REFLECTION_SYSTEM."
                )
            })

    # ── GAP-7: Tool performance tidak dicatat ────────────────────────────────
    if "tool_performance" not in content and "tool_call_count" in content:
        results.append({
            "id": "GAP-7", "severity": "LOW",
            "title": "Performa per-tool tidak dicatat ke DB",
            "detail": (
                "tool_call_count hanya in-memory dan hilang saat restart.\n"
                "Tidak ada data tentang tools mana yang sering gagal.\n"
                "FIX: Log hasil setiap jalankan_tool() ke tabel tool_performance."
            )
        })

    # ── GAP-8: Max iterations terlalu rendah ─────────────────────────────────
    max_iter_match = re.search(r"MAX_ITERATIONS\s*=\s*(\d+)", content)
    if max_iter_match and int(max_iter_match.group(1)) < 10:
        results.append({
            "id": "GAP-8", "severity": "LOW",
            "title": f"MAX_ITERATIONS={max_iter_match.group(1)} mungkin terlalu rendah untuk analisis kompleks",
            "detail": (
                f"Dengan 10 tools wajib + opsional, {max_iter_match.group(1)} iterasi bisa tidak cukup.\n"
                "Analisis parlay multi-match bisa butuh lebih banyak langkah.\n"
                "FIX: Naikkan ke 12-15, atau buat adaptif berdasarkan kompleksitas pertanyaan."
            )
        })

    # ── Hitung severity ───────────────────────────────────────────────────────
    high   = sum(1 for r in results if r["severity"] == "HIGH")
    medium = sum(1 for r in results if r["severity"] == "MEDIUM")
    low    = sum(1 for r in results if r["severity"] == "LOW")

    output = [
        f"🔍 STATIC ANALYSIS REPORT — {TARGET_FILE}",
        f"{'─'*55}",
        f"Ditemukan: {len(results)} masalah (🔴{high} HIGH | 🟡{medium} MEDIUM | 🟢{low} LOW)",
        f"{'─'*55}",
    ]
    for r in results:
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r["severity"], "⚪")
        output.append(f"\n{icon} [{r['id']}] {r['title']}")
        output.append(f"   {r['detail']}")

    return "\n".join(output)


def tool_write_changelog(params_json: str) -> str:
    """
    Tulis ringkasan sesi ke CHANGELOG.
    Input JSON: {"summary": "Sesi X: fix 3 bugs", "improvements": [...]}
    """
    try:
        p = json.loads(params_json)
    except Exception:
        p = {"summary": params_json}

    summary      = p.get("summary", "")
    improvements = p.get("improvements", [])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n\n## [{now}] Coding Agent Session\n\n{summary}\n"
    if improvements:
        entry += "\n### Improvements Applied:\n"
        for imp in improvements:
            entry += f"- **[{imp.get('id', '?')}]** {imp.get('desc', '')}\n"

    try:
        with open(CHANGELOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"✅ Changelog ditulis ke {CHANGELOG_FILE}"
    except Exception as e:
        return f"Error menulis changelog: {e}"


# ─── Helper internal ─────────────────────────────────────────────────────────
def _tulis_changelog(issue_id, desc, old_code, new_code, ts):
    try:
        entry = (
            f"\n\n## [{ts}] Patch [{issue_id}]\n"
            f"**Desc:** {desc}\n\n"
            f"```python\n# BEFORE\n{old_code[:300]}\n```\n\n"
            f"```python\n# AFTER\n{new_code[:300]}\n```\n"
        )
        with open(CHANGELOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_CODING_TOOLS = {
    "read_file":            tool_read_file,
    "search_in_file":       tool_search_in_file,
    "validate_python":      tool_validate_python,
    "apply_patch":          tool_apply_patch,
    "add_prediction":       tool_add_prediction,
    "resolve_prediction":   tool_resolve_prediction,
    "prediction_stats":     tool_prediction_stats,
    "analyze_code":         tool_analyze_code,
    "generate_patch":       tool_generate_patch,
    "run_static_analysis":  tool_run_static_analysis,
    "write_changelog":      tool_write_changelog,
}

def run_coding_tool(name: str, input_str: str) -> str:
    fn = _CODING_TOOLS.get(name.strip())
    if fn is None:
        available = ", ".join(_CODING_TOOLS.keys())
        return f"Error: Tool '{name}' tidak dikenal. Tersedia: {available}"
    try:
        return fn(input_str)
    except Exception as e:
        return f"Error menjalankan tool '{name}': {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — CODING AGENT
# ══════════════════════════════════════════════════════════════════════════════

_CODING_SYSTEM = """
██████████████████████████████████████████████████████████████
CODING AGENT — Football Prediction Optimizer
Mode: ReAct (Reason + Act) | Target: main.py
██████████████████████████████████████████████████████████████

Anda adalah senior AI engineer yang bertugas MENGOPTIMALKAN main.py
(Football Analysis Agent untuk prediksi parlay bola).

OUTPUT FORMAT — Gunakan PERSIS salah satu format ini:

FORMAT A (Tool Call):
Thought: <satu kalimat alasan>
Action: <nama_tool>
Action Input: <JSON input>

FORMAT B (Selesai):
Thought: Semua perbaikan sudah diterapkan.
Final Answer:
<ringkasan lengkap apa saja yang diperbaiki>

TOOLS TERSEDIA:
1. run_static_analysis  — Scan main.py, temukan semua kelemahan (PANGGIL PERTAMA)
2. search_in_file       — Cari fungsi/pattern di main.py
3. read_file            — Baca bagian tertentu dari main.py (by line number)
4. analyze_code         — Minta LLM analisis fungsi tertentu secara mendalam
5. generate_patch       — Buat kode perbaikan untuk issue tertentu
6. validate_python      — Validasi sintaks sebelum apply
7. apply_patch          — Terapkan patch ke main.py (dengan backup otomatis)
8. add_prediction       — Catat prediksi ke tracking DB
9. resolve_prediction   — Update hasil aktual pertandingan
10. prediction_stats    — Lihat akurasi prediksi historis
11. write_changelog     — Tulis ringkasan ke changelog

ALUR KERJA WAJIB:
1. run_static_analysis → identifikasi semua masalah
2. Untuk setiap masalah HIGH severity:
   a. search_in_file   → temukan kode yang bermasalah
   b. generate_patch   → buat perbaikannya
   c. validate_python  → cek sintaks patch
   d. apply_patch      → terapkan patch
3. write_changelog     → catat semua perubahan
4. Final Answer        → ringkasan

PRIORITAS: Selalu perbaiki HIGH severity dulu, lalu MEDIUM, lalu LOW.
Jangan skip validasi sintaks sebelum apply_patch.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN REACT LOOP — CODING AGENT
# ══════════════════════════════════════════════════════════════════════════════

def jalankan_coding_agent(instruksi: str = "Optimalkan main.py untuk prediksi parlay bola.") -> str:
    """
    Jalankan Coding Agent dengan ReAct loop untuk mengoptimalkan main.py.
    """
    session_id = f"coding_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.info("🛠️  Coding Agent dimulai | Session: %s", session_id)

    # Daftarkan sesi ke DB
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR IGNORE INTO coding_sessions (session_id) VALUES (?)",
            (session_id,)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    history = f"Instruksi: {instruksi}\n"
    tool_calls = {}
    final_answer = None

    for iterasi in range(1, MAX_ITERATIONS + 1):
        log.info("─── Iterasi Coding %d/%d ───", iterasi, MAX_ITERATIONS)

        respon = _llm(
            system=_CODING_SYSTEM,
            user=history,
            temperature=0.1
        )
        if not respon:
            log.error("LLM tidak merespons.")
            break

        # ── Final Answer? ─────────────────────────────────────────────────────
        if "Final Answer:" in respon:
            final_answer = respon.split("Final Answer:", 1)[-1].strip()
            log.info("✅ Coding agent selesai di iterasi %d.", iterasi)
            break

        # ── Action? ──────────────────────────────────────────────────────────
        if "Action:" in respon and "Action Input:" in respon:
            try:
                tool_name  = respon.split("Action:", 1)[1].split("Action Input:", 1)[0].strip()
                tool_input = respon.split("Action Input:", 1)[1]
                if "Observation:" in tool_input:
                    tool_input = tool_input.split("Observation:", 1)[0]
                tool_name  = tool_name.splitlines()[0].strip()
                tool_input = tool_input.strip()
            except IndexError:
                history += f"{respon}\nObservation: [parse error]\n"
                continue

            # Guard per-tool
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
            if tool_calls[tool_name] > 3:
                history += (
                    f"Action: {tool_name}\nAction Input: {tool_input}\n"
                    f"Observation: [SISTEM] Tool '{tool_name}' sudah dipanggil 3x. "
                    f"Lanjut ke tool berikutnya atau tulis Final Answer.\n"
                )
                continue

            log.info("🔧 Coding Tool: %-25s | %s", tool_name, tool_input[:60])
            hasil = run_coding_tool(tool_name, tool_input)
            log.info("📋 Hasil: %s…", hasil[:100])

            history += (
                f"Action: {tool_name}\nAction Input: {tool_input}\n"
                f"Observation: {hasil}\n"
            )
            continue

        # ── Tidak ada format yang dikenali ─────────────────────────────────
        log.warning("Format tidak dikenali di iterasi %d. Mendorong...", iterasi)
        n_tools = sum(tool_calls.values())
        if n_tools == 0:
            hint = "WAJIB mulai dengan:\nAction: run_static_analysis\nAction Input: {}"
        else:
            hint = (
                f"Sudah {n_tools} tool calls. Lanjut perbaiki issue berikutnya "
                f"atau tulis Final Answer jika semua selesai."
            )
        history += f"{respon}\nObservation: [SISTEM: {hint}]\n"

    # ── Fallback ──────────────────────────────────────────────────────────────
    if not final_answer:
        log.warning("Loop habis. Meminta ringkasan...")
        final_answer = _llm(
            system=_CODING_SYSTEM,
            user=history + "\n[SISTEM: Loop limit. Tulis Final Answer sekarang.]\n",
            temperature=0.3
        ) or "Sesi coding agent selesai (tanpa final answer eksplisit)."
        if "Final Answer:" in final_answer:
            final_answer = final_answer.split("Final Answer:", 1)[-1].strip()

    # ── Update sesi DB ────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE coding_sessions SET finished_at=? WHERE session_id=?",
            (datetime.now(), session_id)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    log.info("🏁 Coding Agent selesai | Session: %s", session_id)
    return final_answer


# ══════════════════════════════════════════════════════════════════════════════
#  CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def _print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║      🛠️  AGENTIC AI CODING — Football Optimizer          ║
╠══════════════════════════════════════════════════════════╣
║  Commands:                                               ║
║  1. optimize       — Jalankan full optimization loop     ║
║  2. analyze        — Analisis statis main.py saja        ║
║  3. stats          — Lihat akurasi prediksi parlay       ║
║  4. add-result     — Masukkan hasil pertandingan         ║
║  5. exit           — Keluar                              ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    init_coding_db()
    _print_banner()

    if not NVIDIA_API_KEY:
        print("⚠️  WARNING: NVIDIA_API_KEY tidak ditemukan.")
        print("    Set env variable: export NVIDIA_API_KEY=your_key")
        print()

    # ── Mode dari argumen CLI ─────────────────────────────────────────────────
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("Pilih mode (optimize/analyze/stats/add-result): ").strip().lower()

    if mode == "analyze":
        print("\n🔍 Menjalankan Static Analysis...\n")
        print(tool_run_static_analysis("{}"))

    elif mode == "stats":
        print("\n📊 Statistik Prediksi Parlay:\n")
        session = input("Session ID (kosongkan untuk semua): ").strip()
        print(tool_prediction_stats(json.dumps({"session_id": session} if session else {})))

    elif mode == "add-result":
        print("\n📝 Masukkan Hasil Pertandingan:")
        home   = input("Tim Home: ").strip()
        away   = input("Tim Away: ").strip()
        result = input("Hasil Aktual (home/draw/away): ").strip()
        print(tool_resolve_prediction(json.dumps({
            "match_home": home, "match_away": away, "actual_result": result
        })))

    elif mode == "optimize":
        instruksi = input(
            "\nInstruksi optimasi (Enter = default full optimization): "
        ).strip() or "Optimalkan main.py: perbaiki semua bug dan tingkatkan akurasi prediksi parlay."

        print(f"\n🚀 Memulai Coding Agent...\n{'─'*55}\n")
        hasil = jalankan_coding_agent(instruksi)
        print(f"\n{'═'*55}")
        print("✅ CODING AGENT SELESAI")
        print(f"{'═'*55}")
        print(hasil)
        print(f"\nChangelog tersimpan di: {CHANGELOG_FILE}")
        print(f"Backup tersimpan di    : {BACKUP_DIR}/")

    else:
        print(f"Mode '{mode}' tidak dikenal. Gunakan: optimize | analyze | stats | add-result")
