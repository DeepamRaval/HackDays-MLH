"""
snowflake_client.py
===================
Snowflake data layer for TradingAssistant.
Replaces Firebase Firestore for storing stock analyses and chat logs.
All AI features (chat, analysis explanation, history NL→SQL) use
Snowflake Cortex instead of Gemini — Cortex with llama3.1-70b is
available in India (AWS ap-south-1 / Azure Central India regions).

Environment variables required:
    SNOWFLAKE_ACCOUNT   – e.g. xy12345.ap-south-1.aws   (no .snowflakecomputing.com)
    SNOWFLAKE_USER      – your Snowflake login
    SNOWFLAKE_PASSWORD  – your Snowflake password
    SNOWFLAKE_WAREHOUSE – e.g. COMPUTE_WH
    SNOWFLAKE_DATABASE  – e.g. TRADING_DB
    SNOWFLAKE_SCHEMA    – e.g. PUBLIC

Cortex model priority (India-available):
    1. llama3.1-70b   ← primary (best quality available in India)
    2. llama3-70b     ← fallback if 3.1 not yet rolled out to your region
    3. llama3.1-8b    ← fast fallback for latency-sensitive calls
"""

import os
import uuid
import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Optional import — app continues without Snowflake if connector not installed
# ---------------------------------------------------------------------------
try:
    import snowflake.connector
    from snowflake.connector import DictCursor
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("Warning: snowflake-connector-python not installed. Snowflake features disabled.")

# ---------------------------------------------------------------------------
# DDL – tables are created if they don't exist yet
# ---------------------------------------------------------------------------
_DDL_STOCK_ANALYSES = """
CREATE TABLE IF NOT EXISTS STOCK_ANALYSES (
    id            VARCHAR(64)    NOT NULL PRIMARY KEY,
    stock_symbol  VARCHAR(32)    NOT NULL,
    date_from     DATE           NOT NULL,
    date_to       DATE           NOT NULL,
    ema           FLOAT,
    sma           FLOAT,
    rsi           FLOAT,
    volatility    FLOAT,
    support       FLOAT,
    resistance    FLOAT,
    signal        VARCHAR(16),
    trend         VARCHAR(32),
    risk          VARCHAR(32),
    created_at    TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
)
"""

_DDL_CHAT_LOGS = """
CREATE TABLE IF NOT EXISTS CHAT_LOGS (
    id            VARCHAR(64)    NOT NULL PRIMARY KEY,
    session_id    VARCHAR(64),
    user_message  TEXT           NOT NULL,
    ai_response   TEXT,
    created_at    TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
)
"""

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_connection():
    """Return a fresh Snowflake connection using env-var credentials."""
    if not SNOWFLAKE_AVAILABLE:
        raise RuntimeError("snowflake-connector-python is not installed.")

    required = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing Snowflake env vars: {', '.join(missing)}")

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        session_parameters={"QUERY_TAG": "TradingAssistant"},
    )


def _execute(sql, params=None, fetch=False):
    """Run a single statement; return rows if fetch=True."""
    conn = _get_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql, params or ())
            if fetch:
                return cur.fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialisation – call once at app startup
# ---------------------------------------------------------------------------

def init_snowflake():
    """Create tables if they don't exist. Returns True on success."""
    if not SNOWFLAKE_AVAILABLE:
        return False
    try:
        _execute(_DDL_STOCK_ANALYSES)
        _execute(_DDL_CHAT_LOGS)
        print("Snowflake: tables initialised successfully.")
        return True
    except Exception as exc:
        print(f"Warning: Snowflake init failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# STOCK_ANALYSES CRUD
# ---------------------------------------------------------------------------

def save_analysis(stock_symbol: str, date_from: str, date_to: str, result: dict) -> str | None:
    """
    Persist one analysis result row to STOCK_ANALYSES.
    Returns the generated row id, or None on failure.
    """
    if not SNOWFLAKE_AVAILABLE:
        return None
    try:
        row_id = str(uuid.uuid4())
        sql = """
            INSERT INTO STOCK_ANALYSES
                (id, stock_symbol, date_from, date_to,
                 ema, sma, rsi, volatility, support, resistance,
                 signal, trend, risk)
            VALUES
                (%s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s,
                 %s, %s, %s)
        """
        params = (
            row_id,
            stock_symbol.upper(),
            date_from,
            date_to,
            result.get("ema"),
            result.get("sma"),
            result.get("rsi"),
            result.get("volatility"),
            result.get("support"),
            result.get("resistance"),
            result.get("signal"),
            result.get("trend"),
            result.get("risk"),
        )
        _execute(sql, params)
        print(f"Snowflake: analysis saved (id={row_id}, symbol={stock_symbol})")
        return row_id
    except Exception as exc:
        print(f"Warning: could not save analysis to Snowflake: {exc}")
        return None


def get_recent_analyses(limit: int = 20) -> list[dict]:
    """Return the most recent analysis rows."""
    if not SNOWFLAKE_AVAILABLE:
        return []
    try:
        rows = _execute(
            "SELECT * FROM STOCK_ANALYSES ORDER BY created_at DESC LIMIT %s",
            (limit,),
            fetch=True,
        )
        return [_serialize_row(r) for r in (rows or [])]
    except Exception as exc:
        print(f"Warning: could not fetch analyses from Snowflake: {exc}")
        return []


def get_analyses_for_symbol(symbol: str) -> list[dict]:
    """Return all analyses for a given stock symbol."""
    if not SNOWFLAKE_AVAILABLE:
        return []
    try:
        rows = _execute(
            "SELECT * FROM STOCK_ANALYSES WHERE stock_symbol = %s ORDER BY created_at DESC",
            (symbol.upper(),),
            fetch=True,
        )
        return [_serialize_row(r) for r in (rows or [])]
    except Exception as exc:
        print(f"Warning: could not query symbol analyses: {exc}")
        return []


def clear_analyses_history() -> bool:
    """Delete all records from STOCK_ANALYSES."""
    if not SNOWFLAKE_AVAILABLE:
        return False
    try:
        _execute("TRUNCATE TABLE IF EXISTS STOCK_ANALYSES")
        print("Snowflake: STOCK_ANALYSES cleared.")
        return True
    except Exception as exc:
        print(f"Warning: could not clear STOCK_ANALYSES: {exc}")
        return False


# ---------------------------------------------------------------------------
# CHAT_LOGS CRUD
# ---------------------------------------------------------------------------

def save_chat_log(session_id: str, user_message: str, ai_response: str) -> str | None:
    """
    Persist one chat exchange to CHAT_LOGS.
    Returns the generated row id, or None on failure.
    """
    if not SNOWFLAKE_AVAILABLE:
        return None
    try:
        row_id = str(uuid.uuid4())
        sql = """
            INSERT INTO CHAT_LOGS (id, session_id, user_message, ai_response)
            VALUES (%s, %s, %s, %s)
        """
        _execute(sql, (row_id, session_id, user_message, ai_response))
        return row_id
    except Exception as exc:
        print(f"Warning: could not save chat log to Snowflake: {exc}")
        return None


def get_chat_logs(session_id: str | None = None, limit: int = 50) -> list[dict]:
    """Return chat logs, optionally filtered by session."""
    if not SNOWFLAKE_AVAILABLE:
        return []
    try:
        if session_id:
            rows = _execute(
                "SELECT * FROM CHAT_LOGS WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
                (session_id, limit),
                fetch=True,
            )
        else:
            rows = _execute(
                "SELECT * FROM CHAT_LOGS ORDER BY created_at DESC LIMIT %s",
                (limit,),
                fetch=True,
            )
        return [_serialize_row(r) for r in (rows or [])]
    except Exception as exc:
        print(f"Warning: could not fetch chat logs from Snowflake: {exc}")
        return []


# ---------------------------------------------------------------------------
# Cortex – natural-language history query
# ---------------------------------------------------------------------------

_SCHEMA_CONTEXT = """
Tables available:
1. STOCK_ANALYSES
   Columns: id (VARCHAR), stock_symbol (VARCHAR), date_from (DATE), date_to (DATE),
            ema (FLOAT), sma (FLOAT), rsi (FLOAT), volatility (FLOAT),
            support (FLOAT), resistance (FLOAT), signal (VARCHAR),
            trend (VARCHAR), risk (VARCHAR), created_at (TIMESTAMP_NTZ)

2. CHAT_LOGS
   Columns: id (VARCHAR), session_id (VARCHAR), user_message (TEXT),
            ai_response (TEXT), created_at (TIMESTAMP_NTZ)
"""

# llama3.1-70b is the best model available in India (AWS ap-south-1 / Azure Central India).
# mistral-large2 is NOT available in India — do not use it.
_CORTEX_MODEL         = "llama3.1-70b"
_CORTEX_MODEL_FAST    = "llama3.1-8b"       # low-latency tasks
_CORTEX_MODEL_HISTORY = "llama3.1-70b"      # history NL→SQL

# ---------------------------------------------------------------------------
# Public Cortex helper – used by backend.py for chat + analysis explanation
# ---------------------------------------------------------------------------

def cortex_complete(prompt: str, model: str | None = None, system: str | None = None) -> str:
    """
    Call SNOWFLAKE.CORTEX.COMPLETE with the given prompt.
    """
    if not SNOWFLAKE_AVAILABLE:
        raise RuntimeError("Snowflake connector is not installed.")

    chosen_model = model or _CORTEX_MODEL
    conn = _get_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            if system:
                full_prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
            else:
                full_prompt = prompt

            cur.execute(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS result",
                (chosen_model, full_prompt),
            )
            row = cur.fetchone()
            if not row:
                return ""
            val = row.get("RESULT") or row.get("result")
            return str(val).strip() if val else ""
    finally:
        conn.close()


def cortex_complete_safe(prompt: str, system: str | None = None, default: str = "") -> str:
    """
    Like cortex_complete() but never raises — returns `default` on any error.
    Use for optional enrichment (e.g. AI explanation) that must not crash the app.
    """
    try:
        return cortex_complete(prompt, system=system)
    except Exception as exc:
        print(f"cortex_complete_safe: non-critical failure — {exc}")
        return default


def cortex_ask_history(question: str) -> dict:
    """
    Use Snowflake Cortex to translate a natural-language question into SQL,
    execute it, then summarise the result in plain English.
    """
    if not SNOWFLAKE_AVAILABLE:
        return {"error": "Snowflake connector not available."}

    conn = _get_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            sql_gen_prompt = (
                f"[INST] <<SYS>>\n"
                f"You are a SQL generator for Snowflake. Given the schema:\n"
                f"{_SCHEMA_CONTEXT}\n"
                f"IMPORTANT QUERY RULES:\n"
                f"1. When querying STOCK_ANALYSES, ALWAYS SELECT ALL COLUMNS (`SELECT * FROM STOCK_ANALYSES ...`) so that the user interface receives the full stock record (dates, RSI, EMA, SMA, signal, risk, etc.). Never select just a single column like `SELECT stock_symbol`.\n"
                f"2. Symbols in the database often have exchange suffixes (e.g., 'RELIANCE.NS', 'TCS.NS', 'INFY.NS'). When filtering by stock symbol or company name, use case-insensitive partial match: `stock_symbol ILIKE '%RELIANCE%'`.\n"
                f"3. When filtering by signal, use case-insensitive matching: `UPPER(signal) = 'BUY'` or `UPPER(signal) = 'SELL'`.\n\n"
                f"Write ONLY a single valid Snowflake SELECT statement. "
                f"Output only the SQL code, nothing else (no explanations, no commentary, no markdown ticks).\n"
                f"<</SYS>>\n\n"
                f"Generate a SQL query for: \"{question}\" [/INST]"
            )

            cur.execute(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS result",
                (_CORTEX_MODEL, sql_gen_prompt),
            )
            row = cur.fetchone()
            raw_sql = (row.get("RESULT") or row.get("result") or "").strip()

            # Clean markdown formatting if present
            if "```" in raw_sql:
                lines = []
                for line in raw_sql.splitlines():
                    clean_l = line.strip()
                    if clean_l.startswith("```"):
                        continue
                    lines.append(line)
                raw_sql = "\n".join(lines).strip()

            # Remove trailing semicolon if present
            raw_sql = raw_sql.rstrip(";")

            # ── Step 2: safety validation ─────────────────────────────────
            validated_sql = _validate_select(raw_sql)

            # ── Step 3: execute ───────────────────────────────────────────
            cur.execute(validated_sql)
            result_rows = cur.fetchall() or []
            serialised = [_serialize_row(r) for r in result_rows]

            # ── Step 4: summarise via Cortex (best-effort) ────────────────
            answer = _cortex_summarise(cur, question, serialised)

        return {
            "sql": validated_sql,
            "rows": serialised,
            "answer": answer,
        }
    except _SafetyError as e:
        return {"error": f"Query blocked by safety check: {e}", "sql": raw_sql, "rows": [], "answer": ""}
    except Exception as exc:
        print(f"cortex_ask_history error: {exc}")
        return {"error": str(exc), "sql": "", "rows": [], "answer": ""}
    finally:
        conn.close()


class _SafetyError(ValueError):
    pass


_ALLOWED_TABLES = {"STOCK_ANALYSES", "CHAT_LOGS"}


def _validate_select(sql: str) -> str:
    """
    Lightweight safety gate:
    – must start with SELECT
    – must not contain DML / DDL statement keywords as standalone tokens
    """
    import re
    cleaned = sql.strip()
    upper = cleaned.upper()

    if not upper.startswith("SELECT"):
        # If model returned text before SELECT, try to extract the SELECT query
        match = re.search(r"\bSELECT\b[\s\S]+", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(0).strip()
            upper = cleaned.upper()
        else:
            raise _SafetyError("Generated query is not a SELECT statement.")

    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "MERGE", "GRANT", "REVOKE"]
    for kw in forbidden_keywords:
        # Match whole word only (so CREATED_AT or INCREMENT is not blocked)
        if re.search(rf"\b{kw}\b", upper):
            raise _SafetyError(f"Forbidden keyword '{kw}' found in generated SQL.")

    return cleaned


def _cortex_summarise(cur, question: str, rows: list[dict]) -> str:
    """Ask Cortex to turn raw rows into a visually pleasing, structured plain-English answer."""
    try:
        if not rows:
            return "No historical records were found matching your inquiry."

        rows_preview = json.dumps(rows[:10], default=str)
        summary_prompt = (
            f"[INST] <<SYS>>\n"
            f"You are a helpful financial data analyst. Present query answers in a clean, human-readable format. "
            f"Format lists as separate bullet points on new lines. Do NOT write one giant wall of text or run-on sentence. "
            f"Highlight key data like Stock, Date Range, Signal, and RSI clearly.\n"
            f"<</SYS>>\n\n"
            f"User Question: \"{question}\"\n"
            f"Database Records: {rows_preview}\n\n"
            f"Provide a clear, cleanly spaced summary of these results (use bullet points if listing multiple items): [/INST]"
        )
        cur.execute(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS result",
            (_CORTEX_MODEL, summary_prompt),
        )
        row = cur.fetchone()
        val = row.get("RESULT") or row.get("result")
        return str(val).strip() if val else f"Found {len(rows)} matching record(s)."
    except Exception as exc:
        print(f"Cortex summarise failed (non-critical): {exc}")
        return f"{len(rows)} row(s) returned."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialize_row(row: dict) -> dict:
    """Make a Snowflake result row JSON-serialisable."""
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
