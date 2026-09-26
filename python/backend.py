import os
import uuid
from flask import Flask, render_template, request, jsonify, redirect, session
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Load .env if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not available, CORS disabled")

# Snowflake – data layer + Cortex AI (replaces both Firebase and Gemini)
try:
    from python.snowflake_client import (
        init_snowflake,
        save_analysis,
        save_chat_log,
        get_recent_analyses,
        clear_analyses_history,
        cortex_ask_history,
        cortex_complete,
        cortex_complete_safe,
        SNOWFLAKE_AVAILABLE,
    )
    _SNOWFLAKE_CLIENT_OK = True
except ImportError:
    try:
        from snowflake_client import (
            init_snowflake,
            save_analysis,
            save_chat_log,
            get_recent_analyses,
            clear_analyses_history,
            cortex_ask_history,
            cortex_complete,
            cortex_complete_safe,
            SNOWFLAKE_AVAILABLE,
        )
        _SNOWFLAKE_CLIENT_OK = True
    except ImportError:
        _SNOWFLAKE_CLIENT_OK = False
        SNOWFLAKE_AVAILABLE = False
        def init_snowflake(): return False
        def save_analysis(*a, **kw): return None
        def save_chat_log(*a, **kw): return None
        def get_recent_analyses(*a, **kw): return []
        def clear_analyses_history(): return False
        def cortex_ask_history(q): return {"error": "Snowflake client not loaded"}
        def cortex_complete(p, **kw): raise RuntimeError("Snowflake client not loaded")
        def cortex_complete_safe(p, **kw): return ""
        print("Warning: snowflake_client module not found")

# Python analysis engine (unchanged)
try:
    from python.engine import (
        calculate_volatility,
        calculate_sma,
        calculate_ema,
        calculate_rsi,
        find_support_resistance,
    )
except ImportError:
    try:
        from engine import (
            calculate_volatility,
            calculate_sma,
            calculate_ema,
            calculate_rsi,
            find_support_resistance,
        )
    except ImportError as e:
        print(f"CRITICAL: Could not import engine module: {e}")
        raise

# ---------------------------------------------------------------------------
# Initialise services
# ---------------------------------------------------------------------------

# Snowflake – create tables once at startup
SNOWFLAKE_READY = init_snowflake()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
if CORS_AVAILABLE:
    CORS(app)

# System prompt that frames Cortex as a trading assistant
_TRADING_SYSTEM_PROMPT = (
    "You are an expert equity research and technical trading assistant for MLHTrade. "
    "You specialise in technical analysis — EMA, SMA, RSI, volatility, "
    "support/resistance levels, candlestick patterns, and trading setups. "
    "Format your responses cleanly using structured bullet points, clear paragraph breaks, and bold labels. "
    "Avoid giant walls of text. Be direct, clear, and educational. "
    "Do not provide personalised financial advice or guarantee profits."
)


@app.errorhandler(500)
def internal_error(error):
    print(f"500 Error: {error}")
    import traceback
    traceback.print_exc()
    try:
        return render_template("index.html", error="An internal server error occurred. Please try again."), 500
    except Exception:
        return "Internal Server Error", 500


# ---------------------------------------------------------------------------
# Core analysis logic
# ---------------------------------------------------------------------------

def analyze_stock(stock_symbol, date_from, date_to):
    """Analyze stock using Python engine and yfinance, then enrich with Cortex AI."""
    try:
        print(f"=== Starting analysis for {stock_symbol} from {date_from} to {date_to} ===")

        # Validate dates strictly
        try:
            date_from_obj = datetime.strptime(date_from, "%Y-%m-%d").date()
            date_to_obj   = datetime.strptime(date_to,   "%Y-%m-%d").date()
            today = date.today()

            if date_from_obj > today or date_to_obj > today:
                return {"error": "Future dates are not allowed. Please enter a valid date range up to today."}

            if date_from_obj > date_to_obj:
                return {"error": "Start date cannot be after End date. Please select a valid date range."}

            max_lookback = today - timedelta(days=365 * 10)
            if date_from_obj < max_lookback:
                return {"error": "Start date is too far in the past (maximum 10 years lookback)."}

            print(f"Using validated dates: {date_from} to {date_to}")
        except ValueError as ve:
            print(f"ERROR: Invalid date format: {ve}")
            return {"error": f"Invalid date format: {ve}. Please use YYYY-MM-DD."}

        # Stock symbol aliases & benchmark indexes
        cleaned_sym = stock_symbol.strip().upper()
        INDEX_ALIASES = {
            "NSE": "^NSEI",
            "NIFTY": "^NSEI",
            "NIFTY50": "^NSEI",
            "NIFTY 50": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "NIFTYBANK": "^NSEBANK",
            "BANK NIFTY": "^NSEBANK",
            "SENSEX": "^BSESN",
            "BSE": "^BSESN",
        }

        if cleaned_sym in INDEX_ALIASES:
            primary_symbol = INDEX_ALIASES[cleaned_sym]
            candidates = [primary_symbol]
        elif "." in cleaned_sym or cleaned_sym.startswith("^"):
            candidates = [cleaned_sym]
        else:
            # Try NSE first, then BSE, then raw symbol (e.g. US stock or index)
            candidates = [f"{cleaned_sym}.NS", f"{cleaned_sym}.BO", cleaned_sym]

        # Download data safely with candidates
        data = None
        used_symbol = candidates[0]
        for candidate in candidates:
            try:
                print(f"Attempting download for: {candidate}")
                df = yf.download(candidate, start=date_from, end=date_to, progress=False, threads=False)
                if df is not None and not df.empty and len(df) > 0:
                    data = df
                    used_symbol = candidate
                    print(f"Successfully downloaded {len(df)} rows using symbol: {candidate}")
                    break
            except Exception as e:
                print(f"Candidate {candidate} failed: {e}")

        if data is None or data.empty:
            print(f"ERROR: No data available for {stock_symbol}")
            return {"error": f"No market data available for '{stock_symbol}' in the selected date range. If this is a newly listed stock, ensure the start date is after its listing date, or check the symbol ticker."}

        original_symbol = stock_symbol.upper()

        # Extract Close prices
        if "Close" in data.columns:
            prices = data["Close"].dropna().to_numpy(dtype=np.float64)
        else:
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
            if len(numeric_cols) > 0:
                prices = data[numeric_cols[0]].dropna().to_numpy(dtype=np.float64)
            else:
                print(f"ERROR: Could not find Close prices for {original_symbol}")
                return None

        if len(prices) == 0:
            return None

        print(f"Prices: {len(prices)} pts, {prices.min():.2f}–{prices.max():.2f}")

        # Technical indicators
        vol  = calculate_volatility(prices)
        sma  = calculate_sma(prices)
        ema  = calculate_ema(prices, alpha=0.1)
        rsi  = calculate_rsi(prices)
        supports, resistances = find_support_resistance(prices)
        support_level    = supports.min()    if len(supports)    > 0 else prices.min()
        resistance_level = resistances.max() if len(resistances) > 0 else prices.max()

        # Signals
        trend = (
            "bullish" if ema > 1.01 * sma else
            "bearish" if ema < 0.99 * sma else
            "neutral"
        )
        momentum_status = (
            "Overbought" if rsi > 70 else
            "Oversold"   if rsi < 30 else
            "neutral"
        )
        risk_ratio = vol / sma if sma > 0 else 0
        risk = (
            "High Risk"   if risk_ratio > 0.08 else
            "Medium Risk" if risk_ratio > 0.04 else
            "Low Risk"
        )
        final_signal = (
            "BUY"  if trend == "bullish" and rsi < 60 and risk != "High Risk" else
            "SELL" if trend == "bearish" and rsi > 40 else
            "HOLD"
        )

        current_price = prices[-1].item() if hasattr(prices[-1], "item") else float(prices[-1])

        # ── Cortex AI explanation (Snowflake llama3.1-70b) ──────────────────
        explanation_prompt = (
            f"Provide an institutional technical analysis synthesis for {original_symbol} ({date_from} to {date_to}):\n\n"
            f"Metrics Observed:\n"
            f"• Current Price: ₹{round(current_price, 2)}\n"
            f"• EMA (0.1): ₹{round(float(ema), 2)} vs SMA: ₹{round(float(sma), 2)}\n"
            f"• RSI: {round(float(rsi), 2)} ({momentum_status})\n"
            f"• Volatility: {round((float(vol)/float(sma)*100) if sma > 0 else 0, 2)}%\n"
            f"• Support: ₹{round(float(support_level), 2)} | Resistance: ₹{round(float(resistance_level), 2)}\n"
            f"• Signal Computed: {final_signal} | Trend: {trend} | Risk: {risk}\n\n"
            f"Instructions:\n"
            f"1. Explain the rationale behind the {final_signal} signal in 2-3 clear sentences.\n"
            f"2. Add 2 concise bullet points highlighting key price levels to watch (Support/Resistance) and momentum risks.\n"
            f"Format cleanly with bold headings and bullet points."
        )
        ai_explanation = cortex_complete_safe(
            explanation_prompt,
            system=_TRADING_SYSTEM_PROMPT,
            default=None,
        ) or None

        # OHLC chart data (last 90 trading days)
        window = min(90, len(prices))
        chart_dates = [d.strftime("%Y-%m-%d") for d in data.index[-window:]]
        chart_ohlc = []
        for i in range(-window, 0):
            o  = data["Open"].iloc[i].item()  if "Open"  in data.columns else prices[i].item()
            h  = data["High"].iloc[i].item()  if "High"  in data.columns else prices[i].item()
            lo = data["Low"].iloc[i].item()   if "Low"   in data.columns else prices[i].item()
            c  = data["Close"].iloc[i].item() if "Close" in data.columns else prices[i].item()
            chart_ohlc.append({"x": chart_dates[i + window], "o": round(o,2), "h": round(h,2), "l": round(lo,2), "c": round(c,2)})

        result = {
            "volatility":         round(float(vol), 2),
            "volatility_percent": round((float(vol)/float(sma)*100) if sma > 0 else 0, 2),
            "sma":                round(float(sma), 2),
            "ema":                round(float(ema), 2),
            "rsi":                round(float(rsi), 2),
            "support":            round(float(support_level), 2),
            "resistance":         round(float(resistance_level), 2),
            "trend":              trend,
            "momentum":           momentum_status,
            "risk":               risk,
            "signal":             final_signal,
            "current_price":      round(current_price, 2),
            "ai_explanation":     ai_explanation,
            "chart":              {"dates": chart_dates, "ohlc": chart_ohlc},
        }

        # Persist to Snowflake
        save_analysis(original_symbol, date_from, date_to, result)
        return result

    except Exception as e:
        print(f"CRITICAL ERROR analyzing {stock_symbol}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "snowflake": SNOWFLAKE_READY,
        "cortex_ai": SNOWFLAKE_AVAILABLE,   # Cortex is the AI engine now
    }), 200


@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/analysis")
def analysis():
    return render_template("result.html")


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/add_expense", methods=["POST"])
def add_expense():
    stock_name    = request.form.get("stockName", "").strip()
    date_from_str = request.form.get("dateFrom",  "").strip()
    date_to_str   = request.form.get("dateTo",    "").strip()
    if not stock_name or not date_from_str or not date_to_str:
        return render_template("index.html", error="Please fill in all fields"), 200

    try:
        df = datetime.strptime(date_from_str, "%Y-%m-%d").date()
        dt = datetime.strptime(date_to_str, "%Y-%m-%d").date()
        today = date.today()
        if df > today or dt > today:
            return render_template("index.html", error="Future dates are not allowed. Please enter a valid date range up to today."), 200
        if df > dt:
            return render_template("index.html", error="Start date cannot be after End date. Please select a valid date range."), 200
    except ValueError:
        return render_template("index.html", error="Invalid date format. Please use YYYY-MM-DD."), 200

    from urllib.parse import quote
    url = (
        f"/analysis?stock={quote(stock_name)}"
        f"&date_from={quote(date_from_str)}"
        f"&date_to={quote(date_to_str)}"
    )
    return redirect(url, code=302)


@app.route("/api/analyze", methods=["GET", "POST"])
def api_analyze():
    try:
        if request.method == "POST":
            body         = request.get_json() or {}
            stock_symbol = body.get("stock")    or request.form.get("stock")
            date_from    = body.get("date_from") or request.form.get("dateFrom")
            date_to      = body.get("date_to")   or request.form.get("dateTo")
        else:
            stock_symbol = request.args.get("stock")
            date_from    = request.args.get("date_from")
            date_to      = request.args.get("date_to")

        if not stock_symbol or not date_from or not date_to:
            return jsonify({"error": "stock, date_from, and date_to are required"}), 400

        result = analyze_stock(stock_symbol, date_from, date_to)
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400

        if result is None:
            return jsonify({"error": f"Failed to analyze stock '{stock_symbol}'. Please verify the symbol."}), 400

        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Cortex-powered (llama3.1-70b) trading Q&A chat.
    Replaces the previous Gemini endpoint — same JSON contract.
    POST body: { "message": "...", "history": [...] }   (history is optional)
    """
    if not SNOWFLAKE_AVAILABLE:
        return jsonify({"error": "AI chat is not available. Configure SNOWFLAKE_* env vars."}), 503

    body    = request.get_json() or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    # Session ID for chat grouping in CHAT_LOGS
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]

    try:
        response_text = cortex_complete(message, system=_TRADING_SYSTEM_PROMPT)
        if not response_text:
            return jsonify({"error": "Empty response from Cortex"}), 500

        # Persist exchange to Snowflake
        save_chat_log(session_id, message, response_text)

        return jsonify({"response": response_text})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Cortex error: {str(e)}"}), 500


@app.route("/api/ask_history", methods=["POST"])
def api_ask_history():
    """
    Cortex NL→SQL history query.
    POST body: { "question": "..." }
    """
    if not SNOWFLAKE_AVAILABLE:
        return jsonify({"error": "Snowflake not configured. Set SNOWFLAKE_* env vars."}), 503

    body     = request.get_json() or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    result = cortex_ask_history(question)
    if "error" in result and result["error"]:
        return jsonify(result), 422
    return jsonify(result)


@app.route("/api/analyses", methods=["GET"])
def api_analyses():
    """Return recent analyses from Snowflake (for history page)."""
    rows = get_recent_analyses(limit=50)
    return jsonify({"analyses": rows})


@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    """Wipe all analysis history from Snowflake."""
    if not SNOWFLAKE_AVAILABLE:
        return jsonify({"error": "Snowflake is not available."}), 503
    ok = clear_analyses_history()
    if ok:
        return jsonify({"success": True, "message": "History cleared successfully."})
    else:
        return jsonify({"error": "Failed to clear history from database."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(debug=True, host="0.0.0.0", port=port)
