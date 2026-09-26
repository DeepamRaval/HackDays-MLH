# MLHTrade - AI-Powered Trading Intelligence & Technical Analytics

A comprehensive market intelligence platform integrating **Snowflake Cortex AI**, **Google Gemini AI**, and high-precision financial technical analysis engines for Indian and Global Equities.

---

## 🌟 Key Features

- **Smart Technical Analysis**: 
  - Dynamic Candlestick charting with real-time hovering HUD (Open, High, Low, Close, % Return).
  - Technical indicators: Exponential Moving Average (EMA 20), Simple Moving Average (SMA 50), Relative Strength Index (RSI 14), Volatility, Support / Resistance levels.
  - Automated Algorithmic Signal generation (**BUY**, **SELL**, or **HOLD**) with pattern breakdowns.
  - Interactive multi-stage radar progress animations for market computations.
- **AI Market Assistant**:
  - Powered by **Snowflake Cortex LLMs** (`snowflake-arctic`, `llama3.1-70b`) with **Gemini AI** fallback.
  - Formatted markdown tables, structured financial guidance, bulleted risk points, and intelligent analysis.
- **Enterprise Query Engine & Audit History**:
  - Secure Snowflake analytical querying capabilities.
  - Session history viewer with individual deletion, bulk clear, and AI conversational intelligence summaries.
- **Humanized, Modern Dark UI**:
  - Built with Tailwind CSS, Chart.js, Lucide Icons, and typography tuned for trader focus.

---

## 🏗️ Project Architecture

```
HackDays-MLH/
├── python/
│   ├── backend.py            # Flask backend application & API routes
│   ├── engine.py             # Quantitative analysis engine (EMA, SMA, RSI, Signals)
│   ├── snowflake_client.py   # Snowflake Cortex & SQL query client
│   ├── geminiModel.py         # Google Gemini integration client
│   └── templates/
│       ├── index.html        # MLHTrade landing & analysis launcher
│       ├── result.html       # Smart Technical Analysis & Candlestick dashboard
│       ├── chat.html         # AI Market Chat Assistant
│       └── history.html      # Analysis audit history & intelligence summary
├── .env.example              # Template environment variables
├── .gitignore                # Production ignore rules
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/DeepamRaval/HackDays-MLH.git
cd HackDays-MLH
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Key environment variables:
- `SNOWFLAKE_ACCOUNT`: Your Snowflake account identifier
- `SNOWFLAKE_USER`: Snowflake username
- `SNOWFLAKE_PASSWORD`: Snowflake password
- `SNOWFLAKE_DATABASE`: Database name
- `SNOWFLAKE_SCHEMA`: Schema name
- `SNOWFLAKE_WAREHOUSE`: Compute warehouse
- `SNOWFLAKE_ROLE`: Role name
- `GEMINI_API_KEY`: Google Gemini API key

*(Firebase `serviceKey.json` is optional for secondary Firestore storage).*

### 4. Run the Application
```bash
python python/backend.py
```
Visit `http://localhost:5000` (or `http://localhost:5002` if port 5000 is occupied).

---

## 📡 API Endpoints

- `GET /` - MLHTrade Home & Analysis Input
- `GET /analysis` - Smart Technical Analysis Dashboard
- `GET /chat` - Interactive AI Market Assistant
- `GET /history` - Analysis History & Cortex Summary
- `GET /api/analyze?stock=SYMBOL&date_from=YYYY-MM-DD&date_to=YYYY-MM-DD` - Compute indicators and signals
- `POST /api/chat` - Chat with Snowflake Cortex / Gemini AI
- `POST /api/history/delete` - Remove individual or clear all history records

---

## 🛡️ License & Disclaimers
This project was developed for HackDays MLH. Stock market investments are subject to market risks; all AI analysis and algorithmic signals are for informational and hackathon evaluation purposes.
